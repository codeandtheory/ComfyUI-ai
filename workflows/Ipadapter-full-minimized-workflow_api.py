import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        if args is None or args.comfyui_directory is None:
            path = os.getcwd()
        else:
            path = args.comfyui_directory

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


def save_image_wrapper(context, cls):
    if args.output is None:
        return cls

    from PIL import Image, ImageOps, ImageSequence
    from PIL.PngImagePlugin import PngInfo

    import numpy as np

    class WrappedSaveImage(cls):
        counter = 0

        def save_images(
            self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
        ):
            if args.output is None:
                return super().save_images(
                    images, filename_prefix, prompt, extra_pnginfo
                )
            else:
                if len(images) > 1 and args.output == "-":
                    raise ValueError("Cannot save multiple images to stdout")
                filename_prefix += self.prefix_append

                results = list()
                for batch_number, image in enumerate(images):
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    metadata = None
                    if not args.disable_metadata:
                        metadata = PngInfo()
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                    if args.output == "-":
                        # Hack to briefly restore stdout
                        if context is not None:
                            context.__exit__(None, None, None)
                        try:
                            img.save(
                                sys.stdout.buffer,
                                format="png",
                                pnginfo=metadata,
                                compress_level=self.compress_level,
                            )
                        finally:
                            if context is not None:
                                context.__enter__()
                    else:
                        subfolder = ""
                        if len(images) == 1:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = "output.png"
                            else:
                                subfolder, file = os.path.split(args.output)
                                if subfolder == "":
                                    subfolder = os.getcwd()
                        else:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = filename_prefix
                            else:
                                subfolder, file = os.path.split(args.output)

                            if subfolder == "":
                                subfolder = os.getcwd()

                            files = os.listdir(subfolder)
                            file_pattern = file
                            while True:
                                filename_with_batch_num = file_pattern.replace(
                                    "%batch_num%", str(batch_number)
                                )
                                file = (
                                    f"{filename_with_batch_num}_{self.counter:05}.png"
                                )
                                self.counter += 1

                                if file not in files:
                                    break

                        img.save(
                            os.path.join(subfolder, file),
                            pnginfo=metadata,
                            compress_level=self.compress_level,
                        )
                        print("Saved image to", os.path.join(subfolder, file))
                        results.append(
                            {
                                "filename": file,
                                "subfolder": subfolder,
                                "type": self.type,
                            }
                        )

                return {"ui": {"images": results}}

    return WrappedSaveImage


def parse_arg(s: Any):
    """Parses a JSON string, returning it unchanged if the parsing fails."""
    if __name__ == "__main__" or not isinstance(s, str):
        return s

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


parser = argparse.ArgumentParser(
    description="A converted ComfyUI workflow. Required inputs listed below. Values passed should be valid JSON (assumes string if not valid JSON)."
)
parser.add_argument(
    "--queue-size",
    "-q",
    type=int,
    default=1,
    help="How many times the workflow will be executed (default: 1)",
)

# Grounding Dino prompt
parser.add_argument(
    "--gd_prompt",
    "-gdp",
    type=str,
    default="shirt",
    help="Prompt for GroundingDino segmenter",
)

parser.add_argument(
    "--comfyui-directory",
    "-c",
    default=None,
    help="Where to look for ComfyUI (default: current directory)",
)

parser.add_argument(
    "--output",
    "-o",
    default=None,
    help="The location to save the output image. Either a file path, a directory, or - for stdout (default: the ComfyUI output directory)",
)

parser.add_argument(
    "--disable-metadata",
    action="store_true",
    help="Disables writing workflow metadata to the outputs",
)


comfy_args = [sys.argv[0]]
if __name__ == "__main__" and "--" in sys.argv:
    idx = sys.argv.index("--")
    comfy_args += sys.argv[idx + 1 :]
    sys.argv = sys.argv[:idx]

args = None
if __name__ == "__main__":
    args = parser.parse_args()
    sys.argv = comfy_args
if args is not None and args.output is not None and args.output == "-":
    ctx = contextlib.redirect_stdout(sys.stderr)
else:
    ctx = contextlib.nullcontext()

PROMPT_DATA = json.loads(
    '{"2": {"inputs": {"ckpt_name": "absolutereality_v16.safetensors"}, "class_type": "CheckpointLoaderSimple", "_meta": {"title": "Load Checkpoint"}}, "5": {"inputs": {"stop_at_clip_layer": -1, "clip": ["2", 1]}, "class_type": "CLIPSetLastLayer", "_meta": {"title": "CLIP Set Last Layer"}}, "6": {"inputs": {"seed": 566833279836844}, "class_type": "CR Seed", "_meta": {"title": "\\ud83c\\udf31 CR Seed"}}, "9": {"inputs": {"text": "full sleeve shirt", "clip": ["5", 0]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Prompt)"}}, "10": {"inputs": {"text": "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), acne, shiny", "clip": ["5", 0]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Prompt)"}}, "11": {"inputs": {"detect_hand": "enable", "detect_body": "enable", "detect_face": "enable", "resolution": 512, "bbox_detector": "yolox_l.onnx", "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt", "image": ["12", 0]}, "class_type": "DWPreprocessor", "_meta": {"title": "DWPose Estimator"}}, "12": {"inputs": {"image": "0001_m_u_s.png", "upload": "image"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}}, "13": {"inputs": {"control_net_name": "control_v11p_sd15_openpose_fp16.safetensors"}, "class_type": "ControlNetLoader", "_meta": {"title": "Load ControlNet Model"}}, "17": {"inputs": {"strength": 0.5, "start_percent": 0, "end_percent": 0.5, "positive": ["9", 0], "negative": ["10", 0], "control_net": ["13", 0], "image": ["11", 0], "model_optional": ["2", 0]}, "class_type": "ACN_AdvancedControlNetApply", "_meta": {"title": "Apply Advanced ControlNet \\ud83d\\udec2\\ud83c\\udd50\\ud83c\\udd52\\ud83c\\udd5d"}}, "23": {"inputs": {"preset": "PLUS (high strength)", "model": ["17", 2]}, "class_type": "IPAdapterUnifiedLoader", "_meta": {"title": "IPAdapter Unified Loader"}}, "24": {"inputs": {"image": "0001_m_u_s (1).png", "upload": "image"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}}, "25": {"inputs": {"image": "0001_m_u_s (2).png", "upload": "image"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}}, "28": {"inputs": {"weight": 1, "weight_type": "linear", "combine_embeds": "concat", "start_at": 0, "end_at": 1, "embeds_scaling": "K+V", "model": ["23", 0], "ipadapter": ["23", 1], "image": ["24", 0], "attn_mask": ["81", 1]}, "class_type": "IPAdapterAdvanced", "_meta": {"title": "IPAdapter Advanced"}}, "36": {"inputs": {"width": 512, "height": 512, "batch_size": 1}, "class_type": "EmptyLatentImage", "_meta": {"title": "Empty Latent Image"}}, "37": {"inputs": {"seed": 681422518315186, "steps": 30, "cfg": 1.5, "sampler_name": "euler", "scheduler": "karras", "denoise": 1, "model": ["28", 0], "positive": ["17", 0], "negative": ["17", 1], "latent_image": ["36", 0]}, "class_type": "KSampler", "_meta": {"title": "KSampler"}}, "39": {"inputs": {"samples": ["37", 0], "vae": ["2", 2]}, "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"}}, "48": {"inputs": {"seed": 230762624240794, "steps": 30, "cfg": 1.5, "sampler_name": "euler", "scheduler": "karras", "denoise": 1, "model": ["28", 0], "positive": ["17", 0], "negative": ["17", 1], "latent_image": ["37", 0]}, "class_type": "KSampler", "_meta": {"title": "KSampler"}}, "52": {"inputs": {"samples": ["48", 0], "vae": ["2", 2]}, "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"}}, "61": {"inputs": {"model_name": "bbox/face_yolov8m.pt"}, "class_type": "UltralyticsDetectorProvider", "_meta": {"title": "UltralyticsDetectorProvider"}}, "62": {"inputs": {"model_name": "sam_vit_h_4b8939.pth", "device_mode": "AUTO"}, "class_type": "SAMLoader", "_meta": {"title": "SAMLoader (Impact)"}}, "63": {"inputs": {"mode": "768x768"}, "class_type": "CoreMLDetailerHookProvider", "_meta": {"title": "CoreMLDetailerHookProvider"}}, "65": {"inputs": {"model_name": "segm/deepfashion2_yolov8s-seg.pt"}, "class_type": "UltralyticsDetectorProvider", "_meta": {"title": "UltralyticsDetectorProvider"}}, "66": {"inputs": {"model_name": "sam_vit_b_01ec64.pth", "device_mode": "AUTO"}, "class_type": "SAMLoader", "_meta": {"title": "SAMLoader (Impact)"}}, "67": {"inputs": {"mode": "768x768"}, "class_type": "CoreMLDetailerHookProvider", "_meta": {"title": "CoreMLDetailerHookProvider"}}, "68": {"inputs": {"guide_size": 384, "guide_size_for": true, "max_size": 1024, "seed": 923073152666355, "steps": 20, "cfg": 8, "sampler_name": "euler", "scheduler": "normal", "denoise": 0.5, "feather": 5, "noise_mask": true, "force_inpaint": true, "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": 3, "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93, "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "False", "drop_size": 10, "wildcard": "", "cycle": 1, "inpaint_model": false, "noise_mask_feather": 20, "image": ["69", 0], "model": ["28", 0], "clip": ["5", 0], "vae": ["2", 2], "positive": ["17", 0], "negative": ["17", 1], "bbox_detector": ["65", 0], "sam_model_opt": ["66", 0], "segm_detector_opt": ["65", 1], "detailer_hook": ["67", 0]}, "class_type": "FaceDetailer", "_meta": {"title": "FaceDetailer"}}, "69": {"inputs": {"guide_size": 384, "guide_size_for": true, "max_size": 1024, "seed": 821243160793012, "steps": 20, "cfg": 8, "sampler_name": "euler", "scheduler": "normal", "denoise": 0.5, "feather": 5, "noise_mask": true, "force_inpaint": true, "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": 3, "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93, "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "False", "drop_size": 10, "wildcard": "", "cycle": 1, "inpaint_model": false, "noise_mask_feather": 20, "image": ["52", 0], "model": ["28", 0], "clip": ["5", 0], "vae": ["2", 2], "positive": ["17", 0], "negative": ["17", 1], "bbox_detector": ["61", 0], "sam_model_opt": ["62", 0], "segm_detector_opt": ["61", 1], "detailer_hook": ["63", 0]}, "class_type": "FaceDetailer", "_meta": {"title": "FaceDetailer"}}, "71": {"inputs": {"images": ["68", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "72": {"inputs": {"model_name": "8x_NMKD-Superscale_150000_G.pth"}, "class_type": "UpscaleModelLoader", "_meta": {"title": "Load Upscale Model"}}, "73": {"inputs": {"upscale_by": 1.5, "seed": 687565740472152, "steps": 20, "cfg": 8, "sampler_name": "euler", "scheduler": "normal", "denoise": 0.2, "mode_type": "Linear", "tile_width": 512, "tile_height": 512, "mask_blur": 8, "tile_padding": 32, "seam_fix_mode": "None", "seam_fix_denoise": 1, "seam_fix_width": 64, "seam_fix_mask_blur": 8, "seam_fix_padding": 16, "force_uniform_tiles": true, "tiled_decode": false, "image": ["68", 0], "model": ["28", 0], "positive": ["17", 0], "negative": ["17", 1], "vae": ["2", 2], "upscale_model": ["72", 0]}, "class_type": "UltimateSDUpscale", "_meta": {"title": "Ultimate SD Upscale"}}, "74": {"inputs": {"images": ["73", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "75": {"inputs": {"model_name": "4x-UltraSharp.pth"}, "class_type": "UpscaleModelLoader", "_meta": {"title": "Load Upscale Model"}}, "77": {"inputs": {"upscale_model": ["75", 0], "image": ["73", 0]}, "class_type": "ImageUpscaleWithModel", "_meta": {"title": "Upscale Image (using Model)"}}, "79": {"inputs": {"model_name": "sam_vit_h (2.56GB)"}, "class_type": "SAMModelLoader (segment anything)", "_meta": {"title": "SAMModelLoader (segment anything)"}}, "80": {"inputs": {"model_name": "GroundingDINO_SwinB (938MB)"}, "class_type": "GroundingDinoModelLoader (segment anything)", "_meta": {"title": "GroundingDinoModelLoader (segment anything)"}}, "81": {"inputs": {"prompt": "shirt", "threshold": 0.3, "sam_model": ["79", 0], "grounding_dino_model": ["80", 0], "image": ["25", 0]}, "class_type": "GroundingDinoSAMSegment (segment anything)", "_meta": {"title": "GroundingDinoSAMSegment (segment anything)"}}, "85": {"inputs": {"filename_prefix": "ComfyUI", "images": ["77", 0]}, "class_type": "SaveImage", "_meta": {"title": "Save Image"}}}'
)


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


_custom_nodes_imported = False
_custom_path_added = False


def main(*func_args, **func_kwargs):
    global args, _custom_nodes_imported, _custom_path_added
    if __name__ == "__main__":
        if args is None:
            args = parser.parse_args()
    else:
        defaults = dict(
            (arg, parser.get_default(arg))
            for arg in ["queue_size", "comfyui_directory", "output", "disable_metadata"]
        )
        ordered_args = dict(zip([], func_args))

        all_args = dict()
        all_args.update(defaults)
        all_args.update(ordered_args)
        all_args.update(func_kwargs)

        args = argparse.Namespace(**all_args)

    with ctx:
        if not _custom_path_added:
            add_comfyui_directory_to_sys_path()
            add_extra_model_paths()

            _custom_path_added = True

        if not _custom_nodes_imported:
            import_custom_nodes()

            _custom_nodes_imported = True

        from nodes import (
            CheckpointLoaderSimple,
            KSampler,
            LoadImage,
            VAEDecode,
            EmptyLatentImage,
            SaveImage,
            CLIPSetLastLayer,
            ControlNetLoader,
            NODE_CLASS_MAPPINGS,
            CLIPTextEncode,
        )

    with torch.inference_mode(), ctx:
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_2 = checkpointloadersimple.load_checkpoint(
            ckpt_name="absolutereality_v16.safetensors"
        )

        cr_seed = NODE_CLASS_MAPPINGS["CR Seed"]()
        cr_seed_6 = cr_seed.seedint(seed=random.randint(1, 2**64))

        clipsetlastlayer = CLIPSetLastLayer()
        clipsetlastlayer_5 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-1, clip=get_value_at_index(checkpointloadersimple_2, 1)
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_9 = cliptextencode.encode(
            text="full sleeve shirt", clip=get_value_at_index(clipsetlastlayer_5, 0)
        )

        cliptextencode_10 = cliptextencode.encode(
            text="(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), acne, shiny",
            clip=get_value_at_index(clipsetlastlayer_5, 0),
        )

        loadimage = LoadImage()
        loadimage_12 = loadimage.load_image(image="0001_m_u_s.png")

        controlnetloader = ControlNetLoader()
        controlnetloader_13 = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_openpose_fp16.safetensors"
        )

        loadimage_24 = loadimage.load_image(image="0001_m_u_s (1).png")

        loadimage_25 = loadimage.load_image(image="0001_m_u_s (2).png")

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_36 = emptylatentimage.generate(
            width=512, height=512, batch_size=1
        )

        ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS[
            "UltralyticsDetectorProvider"
        ]()
        ultralyticsdetectorprovider_61 = ultralyticsdetectorprovider.doit(
            model_name="bbox/face_yolov8m.pt"
        )

        samloader = NODE_CLASS_MAPPINGS["SAMLoader"]()
        samloader_62 = samloader.load_model(
            model_name="sam_vit_h_4b8939.pth", device_mode="AUTO"
        )

        coremldetailerhookprovider = NODE_CLASS_MAPPINGS["CoreMLDetailerHookProvider"]()
        coremldetailerhookprovider_63 = coremldetailerhookprovider.doit(mode="768x768")

        ultralyticsdetectorprovider_65 = ultralyticsdetectorprovider.doit(
            model_name="segm/deepfashion2_yolov8s-seg.pt"
        )

        samloader_66 = samloader.load_model(
            model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
        )

        coremldetailerhookprovider_67 = coremldetailerhookprovider.doit(mode="768x768")

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_72 = upscalemodelloader.load_model(
            model_name="8x_NMKD-Superscale_150000_G.pth"
        )

        upscalemodelloader_75 = upscalemodelloader.load_model(
            model_name="4x-UltraSharp.pth"
        )

        sammodelloader_segment_anything = NODE_CLASS_MAPPINGS[
            "SAMModelLoader (segment anything)"
        ]()
        sammodelloader_segment_anything_79 = sammodelloader_segment_anything.main(
            model_name="sam_vit_h (2.56GB)"
        )

        groundingdinomodelloader_segment_anything = NODE_CLASS_MAPPINGS[
            "GroundingDinoModelLoader (segment anything)"
        ]()
        groundingdinomodelloader_segment_anything_80 = (
            groundingdinomodelloader_segment_anything.main(
                model_name="GroundingDINO_SwinB (938MB)"
            )
        )

        dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
        acn_advancedcontrolnetapply = NODE_CLASS_MAPPINGS[
            "ACN_AdvancedControlNetApply"
        ]()
        ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
        groundingdinosamsegment_segment_anything = NODE_CLASS_MAPPINGS[
            "GroundingDinoSAMSegment (segment anything)"
        ]()
        ipadapteradvanced = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        facedetailer = NODE_CLASS_MAPPINGS["FaceDetailer"]()
        ultimatesdupscale = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]()
        imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        saveimage = save_image_wrapper(ctx, SaveImage)()
        for q in range(args.queue_size):
            dwpreprocessor_11 = dwpreprocessor.estimate_pose(
                detect_hand="enable",
                detect_body="enable",
                detect_face="enable",
                resolution=512,
                bbox_detector="yolox_l.onnx",
                pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
                image=get_value_at_index(loadimage_12, 0),
            )

            acn_advancedcontrolnetapply_17 = (
                acn_advancedcontrolnetapply.apply_controlnet(
                    strength=0.5,
                    start_percent=0,
                    end_percent=0.5,
                    positive=get_value_at_index(cliptextencode_9, 0),
                    negative=get_value_at_index(cliptextencode_10, 0),
                    control_net=get_value_at_index(controlnetloader_13, 0),
                    image=get_value_at_index(dwpreprocessor_11, 0),
                    model_optional=get_value_at_index(checkpointloadersimple_2, 0),
                )
            )

            ipadapterunifiedloader_23 = ipadapterunifiedloader.load_models(
                preset="PLUS (high strength)",
                model=get_value_at_index(acn_advancedcontrolnetapply_17, 2),
            )

            groundingdinosamsegment_segment_anything_81 = (
                groundingdinosamsegment_segment_anything.main(
                    prompt=args.gd_prompt,
                    threshold=0.3,
                    sam_model=get_value_at_index(sammodelloader_segment_anything_79, 0),
                    grounding_dino_model=get_value_at_index(
                        groundingdinomodelloader_segment_anything_80, 0
                    ),
                    image=get_value_at_index(loadimage_25, 0),
                )
            )

            ipadapteradvanced_28 = ipadapteradvanced.apply_ipadapter(
                weight=1,
                weight_type="linear",
                combine_embeds="concat",
                start_at=0,
                end_at=1,
                embeds_scaling="K+V",
                model=get_value_at_index(ipadapterunifiedloader_23, 0),
                ipadapter=get_value_at_index(ipadapterunifiedloader_23, 1),
                image=get_value_at_index(loadimage_24, 0),
                attn_mask=get_value_at_index(
                    groundingdinosamsegment_segment_anything_81, 1
                ),
            )

            ksampler_37 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=30,
                cfg=1.5,
                sampler_name="euler",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(ipadapteradvanced_28, 0),
                positive=get_value_at_index(acn_advancedcontrolnetapply_17, 0),
                negative=get_value_at_index(acn_advancedcontrolnetapply_17, 1),
                latent_image=get_value_at_index(emptylatentimage_36, 0),
            )

            vaedecode_39 = vaedecode.decode(
                samples=get_value_at_index(ksampler_37, 0),
                vae=get_value_at_index(checkpointloadersimple_2, 2),
            )

            ksampler_48 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=30,
                cfg=1.5,
                sampler_name="euler",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(ipadapteradvanced_28, 0),
                positive=get_value_at_index(acn_advancedcontrolnetapply_17, 0),
                negative=get_value_at_index(acn_advancedcontrolnetapply_17, 1),
                latent_image=get_value_at_index(ksampler_37, 0),
            )

            vaedecode_52 = vaedecode.decode(
                samples=get_value_at_index(ksampler_48, 0),
                vae=get_value_at_index(checkpointloadersimple_2, 2),
            )

            facedetailer_69 = facedetailer.doit(
                guide_size=384,
                guide_size_for=True,
                max_size=1024,
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.5,
                feather=5,
                noise_mask=True,
                force_inpaint=True,
                bbox_threshold=0.5,
                bbox_dilation=10,
                bbox_crop_factor=3,
                sam_detection_hint="center-1",
                sam_dilation=0,
                sam_threshold=0.93,
                sam_bbox_expansion=0,
                sam_mask_hint_threshold=0.7,
                sam_mask_hint_use_negative="False",
                drop_size=10,
                wildcard="",
                cycle=1,
                inpaint_model=False,
                noise_mask_feather=20,
                image=get_value_at_index(vaedecode_52, 0),
                model=get_value_at_index(ipadapteradvanced_28, 0),
                clip=get_value_at_index(clipsetlastlayer_5, 0),
                vae=get_value_at_index(checkpointloadersimple_2, 2),
                positive=get_value_at_index(acn_advancedcontrolnetapply_17, 0),
                negative=get_value_at_index(acn_advancedcontrolnetapply_17, 1),
                bbox_detector=get_value_at_index(ultralyticsdetectorprovider_61, 0),
                sam_model_opt=get_value_at_index(samloader_62, 0),
                segm_detector_opt=get_value_at_index(ultralyticsdetectorprovider_61, 1),
                detailer_hook=get_value_at_index(coremldetailerhookprovider_63, 0),
            )

            facedetailer_68 = facedetailer.doit(
                guide_size=384,
                guide_size_for=True,
                max_size=1024,
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.5,
                feather=5,
                noise_mask=True,
                force_inpaint=True,
                bbox_threshold=0.5,
                bbox_dilation=10,
                bbox_crop_factor=3,
                sam_detection_hint="center-1",
                sam_dilation=0,
                sam_threshold=0.93,
                sam_bbox_expansion=0,
                sam_mask_hint_threshold=0.7,
                sam_mask_hint_use_negative="False",
                drop_size=10,
                wildcard="",
                cycle=1,
                inpaint_model=False,
                noise_mask_feather=20,
                image=get_value_at_index(facedetailer_69, 0),
                model=get_value_at_index(ipadapteradvanced_28, 0),
                clip=get_value_at_index(clipsetlastlayer_5, 0),
                vae=get_value_at_index(checkpointloadersimple_2, 2),
                positive=get_value_at_index(acn_advancedcontrolnetapply_17, 0),
                negative=get_value_at_index(acn_advancedcontrolnetapply_17, 1),
                bbox_detector=get_value_at_index(ultralyticsdetectorprovider_65, 0),
                sam_model_opt=get_value_at_index(samloader_66, 0),
                segm_detector_opt=get_value_at_index(ultralyticsdetectorprovider_65, 1),
                detailer_hook=get_value_at_index(coremldetailerhookprovider_67, 0),
            )

            ultimatesdupscale_73 = ultimatesdupscale.upscale(
                upscale_by=1.5,
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.2,
                mode_type="Linear",
                tile_width=512,
                tile_height=512,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode="None",
                seam_fix_denoise=1,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False,
                image=get_value_at_index(facedetailer_68, 0),
                model=get_value_at_index(ipadapteradvanced_28, 0),
                positive=get_value_at_index(acn_advancedcontrolnetapply_17, 0),
                negative=get_value_at_index(acn_advancedcontrolnetapply_17, 1),
                vae=get_value_at_index(checkpointloadersimple_2, 2),
                upscale_model=get_value_at_index(upscalemodelloader_72, 0),
            )

            imageupscalewithmodel_77 = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscalemodelloader_75, 0),
                image=get_value_at_index(ultimatesdupscale_73, 0),
            )

            if __name__ != "__main__":
                return dict(
                    filename_prefix="ComfyUI",
                    images=get_value_at_index(imageupscalewithmodel_77, 0),
                    prompt=PROMPT_DATA,
                )
            else:
                saveimage_85 = saveimage.save_images(
                    filename_prefix="ComfyUI",
                    images=get_value_at_index(imageupscalewithmodel_77, 0),
                    prompt=PROMPT_DATA,
                )


if __name__ == "__main__":
    main()
