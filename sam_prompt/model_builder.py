# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr
from .model.decoder import TransformerDecoder, TransformerDecoderLayer
from .model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from .model.geometry_encoders import SequenceGeometryEncoder
from .model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from .model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from .model.necks import Sam3DualViTDetNeck
from .model.position_encoding import PositionEmbeddingSine
from .model.sam3_image_processor import Sam3Processor
from .model.text_encoder_ve import VETextEncoder
from .model.tokenizer_ve import SimpleTokenizer
from .model.vitdet import ViT
from .model.vl_combiner import SAM3VLBackbone

from importlib.resources import files, as_file
from PIL import Image


def get_bpe_path() -> str:
    with as_file(
        files("sam_prompt").joinpath("assets/bpe_simple_vocab_16e6.txt.gz")
    ) as p:
        return str(p)


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone():
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=None,
    )


def _create_vit_neck(position_encoding, vit_backbone):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=False,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head():
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=None,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding()
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
):
    """Create the SAM3 image model."""

    from .model.sam3_image import Sam3Image

    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": None,
        "matcher": None,
    }
    model = Sam3Image(**common_params)

    return model


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_vision_backbone() -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    if torch.cuda.is_available():
        position_encoding = _create_position_encoding(precompute_resolution=1008)
    else:
        position_encoding = _create_position_encoding()
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone()
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(position_encoding, vit_backbone)
    # Visual neck
    return vit_neck


def _create_sam3_transformer() -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _load_checkpoint(model, checkpoint_path):
    """Load model checkpoint from file."""
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    model.load_state_dict(sam3_image_ckpt, strict=False)


def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")
    model = model.to(target_device)
    if eval_mode:
        model.eval()
    return model


def build_sam3_image_model(
    *,
    bpe_path: str | None = None,
    checkpoint_or_hf_token: str | None = None,
    device: str | None = None,
):
    """
    Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        checkpoint_or_hf_token: Optional path to model checkpoint or Hugging Face token.
        device: Device to load the model on (for example 'cpu', 'cuda', or 'cuda:0')

    Returns:
        A SAM3 image model
    """
    if bpe_path is None:
        bpe_path = get_bpe_path()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create visual components
    vision_encoder = _create_vision_backbone()

    # Create text components
    text_encoder = _create_text_encoder(bpe_path)

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = _create_segmentation_head()

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
    )

    if not checkpoint_or_hf_token or not checkpoint_or_hf_token.endswith(".pt"):
        checkpoint_path = download_ckpt_from_hf(checkpoint_or_hf_token)
    else:
        checkpoint_path = checkpoint_or_hf_token

    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, True)

    return model


def download_ckpt_from_hf(token: str | None) -> str:
    return hf_hub_download(repo_id="facebook/sam3.1", filename="sam3.1_multiplex.pt")


@lru_cache(maxsize=1)
def build_sam3_image_model_cached(
    *,
    bpe_path: str | None = None,
    checkpoint_or_hf_token: str | None = None,
    device: str | None = None,
):
    return build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_or_hf_token=checkpoint_or_hf_token,
        device=device,
    )


def build_prompt_function(
    image: NDArray | Image.Image | str,
    confidence_threshold=0.5,
    checkpoint_or_hf_token: str | None = None,
):
    model = build_sam3_image_model_cached(checkpoint_or_hf_token=checkpoint_or_hf_token)
    processor = Sam3Processor(
        model, device=model.device.type, confidence_threshold=confidence_threshold
    )
    if isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).to(model.device)
    state = processor.set_image(tensor)

    def set_prompt(prompt: str):
        dictionary = processor.set_text_prompt(prompt, state)
        results: list[tuple[NDArray[np.bool_], float]] = []
        for mask, score in zip(dictionary["masks"], dictionary["scores"], strict=True):
            mask_array = np.squeeze(mask.cpu().numpy())
            results.append((mask_array, float(score)))
        return sorted(results, key=lambda x: x[1], reverse=True)

    return set_prompt
