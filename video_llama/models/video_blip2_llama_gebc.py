import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from video_llama.common.registry import registry
from video_llama.models.blip2 import Blip2Base, disabled_train
from video_llama.models.modeling_llama import LlamaForCausalLM
# from video_llama.models.Qformer import BertEncoder
from transformers import LlamaTokenizer,BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
import copy
from video_llama.models.Qformer import BertConfig, BertLMHeadModel
import math
# from flamingo_pytorch import PerceiverResampler
@registry.register_model("video_blip2_llama_gebc")
class VideoBLIP2LLAMA(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
    }
    
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers =2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    
    def __init__(
        self,
        llama_model="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.

        frozen_llama_proj=True,
        llama_proj_model='',
        max_frame_pos= 32,
        num_video_query_token = 32,
        q_former_hidden_size=768
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.q_former_hidden_size = q_former_hidden_size
        logging.info('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        logging.info('Loading LLAMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLAMA Done')


        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.q_former_hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.llama_proj.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        logging.info('Loading llama_proj Done')

        self.max_txt_len = max_txt_len
        self.end_sym = self.llama_tokenizer.eos_token

        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.q_former_hidden_size)
        self.num_video_query_token = num_video_query_token
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,\
            vision_width=self.q_former_hidden_size, num_hidden_layers =2)

        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.temporal_pos_trans = nn.Linear(self.q_former_hidden_size, self.q_former_hidden_size)
        self.temporal_pos_trans_norm = nn.LayerNorm(self.q_former_hidden_size)


    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = self.q_former_hidden_size / 4
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # batch size, 2
        proposals = proposals.sigmoid() * scale
        # batch size, 2, 128
        pos = proposals[:, :, None] / dim_t
        # batch size, 2, 256
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=4).flatten(2)
        pos = pos.view(pos.shape[0], 1, -1)
        return pos


    def encode_video(self, q_hidden_state, reference_points):
        with self.maybe_autocast():
            # add frame_pos embedding
            batch_size, time_length, _, _ = q_hidden_state.size()
            position_ids = torch.arange(time_length, dtype=torch.long, device=q_hidden_state.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = frame_position_embeddings + q_hidden_state
            # frame attention
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(q_hidden_state.device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            
            # Embed boundary information,  batch size, 1, hidden_size
            reference_point_embed = self.temporal_pos_trans_norm(self.temporal_pos_trans(self.get_proposal_pos_embed(reference_points)))
            video_query_tokens = video_query_tokens + reference_point_embed
            
            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            video_tokens = self.llama_proj(video_hidden)
            video_att_mask = torch.ones(video_tokens.size()[:-1], dtype=torch.long).to(video_tokens.device)
        return video_tokens, video_att_mask
            
            
    def prompt_wrap(self, video_embeds, atts_video, prompt):
        if prompt:
            batch_size = video_embeds.shape[0]
            # print(prompt)
            p_before_tokens = self.llama_tokenizer(
                'Video:', return_tensors="pt", add_special_tokens=False).to(video_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False, padding='longest').to(video_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
            p_after_attention_mask = p_after_tokens.attention_mask
            wrapped_video_embeds = torch.cat([p_before_embeds, video_embeds], dim=1)
            wrapped_atts_video = atts_video[:, :1].expand(-1, wrapped_video_embeds.shape[1])
            wrapped_video_embeds = torch.cat([wrapped_video_embeds, p_after_embeds], dim=1)
            wrapped_atts_video = torch.cat([wrapped_atts_video, p_after_attention_mask], dim=1)
            return wrapped_video_embeds, wrapped_atts_video
        else:
            return video_embeds, atts_video
    
    
    def forward(self, samples):
        image_query_tokens = samples['image_query_tokens']

        reference_points = samples['reference_points']
        video_embeds, atts_video = self.encode_video(image_query_tokens, reference_points)

        prompt = samples['prompt']
        video_embeds, atts_video = self.prompt_wrap(video_embeds, atts_video, prompt)

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(video_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([video_embeds.shape[0], atts_video.shape[1]+1],
                dtype=torch.long).to(video_embeds.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)
            
        batch_size = video_embeds.shape[0]
        bos = torch.ones([batch_size, 1],dtype=to_regress_tokens.input_ids.dtype,
                device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_video[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, video_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_video, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        
        with self.maybe_autocast():
            image_query_tokens = samples['image_query_tokens']
            reference_points = samples['reference_points']
            video_embeds, atts_video = self.encode_video(image_query_tokens, reference_points)

            prompt = samples['prompt']
            video_embeds, atts_video = self.prompt_wrap(video_embeds, atts_video, prompt)

            batch_size = video_embeds.shape[0]
            bos = torch.ones([batch_size, 1], device=video_embeds.device).long() * self.llama_tokenizer.bos_token_id
            bos_embeds = self.opt_model.model.decoder.embed_tokens(bos)
            atts_bos = atts_video[:, :1]
            
            inputs_embeds = torch.cat([bos_embeds, video_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_video], dim=1)
            
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text for text in output_text]

        return output_text
    
    
    @classmethod
    def from_config(cls, cfg):
        q_former_hidden_size = cfg.get('q_former_hidden_size', 768)
        llama_model = cfg.get("llama_model")
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        max_txt_len = cfg.get("max_txt_len", 30)
        end_sym = cfg.get("end_sym", '\n')
        frozen_llama_proj = cfg.get("frozen_llama_proj", False)
        llama_proj_model = cfg.get("llama_proj_model", '')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        num_video_query_token =  cfg.get("num_video_query_token", 32)

        model = cls(
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,  # use 8 bit and put vit in cpu
            device_8bit=device_8bit,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            frozen_llama_proj=frozen_llama_proj,
            llama_proj_model=llama_proj_model,
            max_frame_pos= max_frame_pos,
            num_video_query_token = num_video_query_token,
            q_former_hidden_size=q_former_hidden_size
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model