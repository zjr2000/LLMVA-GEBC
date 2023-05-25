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
        self.end_sym = end_sym

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

    def inverse_sigmoid(self, x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1/x2)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = self.q_former_hidden_size / 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # batch size, 1
        proposals = proposals.sigmoid() * scale
        # batch size, 1, 256
        pos = proposals[:, :, None] / dim_t
        # batch size, 1, 512
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=4).flatten(2)
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
            reference_point_embed = self.get_proposal_pos_embed(self.inverse_sigmoid(reference_points))
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