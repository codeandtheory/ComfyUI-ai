# Requirements for Running IPAdapter

## Download Checkpoints
### save into checkpoints directory:
https://civitai.com/api/download/models/108576
https://civitai.com/api/download/models/329420
https://civitai.com/api/download/models/293240

## Download Clip-Vision Model:
### save into clip_vision directory:
https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/model.safetensors?download=true

## Download ControlNet
### save into controlnet directory
https://huggingface.co/stabilityai/control-lora/blob/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors

## Download Grounding-DINO
### save into grounding-dino directory
https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

## Download IP-Adapter
### save into ipadapter directory
https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors

## Download SAMs
### save into sams directory
https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth?download=true
https://huggingface.co/segments-arnaud/sam_vit_b/resolve/f38484d6934e5d2b555b1685d22d676236455685/sam_vit_b_01ec64.pth?download=true

```python
python3 workflows/IPADPTER_GROUNDINGDINO-workflow_api.py --gd_prompt="shirt" --cloth="0002_m_u_sh_cloth.png" --dmodel="0002_m_u_sh_model.png"
```