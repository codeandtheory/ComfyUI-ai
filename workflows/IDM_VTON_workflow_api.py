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

parser.add_argument(
    "--gd_prompt",
    "-gdp",
    type=str,
    default="shirt",
    help="Prompt for GroundingDino segmenter",
)

parser.add_argument(
    "--cloth",
    "-clo",
    type=str,
    help="Image path for cloth",
)

parser.add_argument(
    "--dmodel",
    "-dm",
    type=str,
    help="Image path for dress model",
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

garment_desc = "a male model wearing a full sleeve tshirt"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

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

        from nodes import LoadImage, SaveImage, NODE_CLASS_MAPPINGS

    with torch.inference_mode(), ctx:
        loadimage = LoadImage()
        loadimage_1 = loadimage.load_image(image=args.dmodel)

        loadimage_2 = loadimage.load_image(image=args.cloth)

        sammodelloader_segment_anything = NODE_CLASS_MAPPINGS[
            "SAMModelLoader (segment anything)"
        ]()
        sammodelloader_segment_anything_4 = sammodelloader_segment_anything.main(
            model_name="sam_vit_h (2.56GB)"
        )

        groundingdinomodelloader_segment_anything = NODE_CLASS_MAPPINGS[
            "GroundingDinoModelLoader (segment anything)"
        ]()
        groundingdinomodelloader_segment_anything_5 = (
            groundingdinomodelloader_segment_anything.main(
                model_name="GroundingDINO_SwinB (938MB)"
            )
        )

        pipelineloader = NODE_CLASS_MAPPINGS["PipelineLoader"]()
        pipelineloader_10 = pipelineloader.load_pipeline(weight_dtype="float16")

        groundingdinosamsegment_segment_anything = NODE_CLASS_MAPPINGS[
            "GroundingDinoSAMSegment (segment anything)"
        ]()
        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        denseposepreprocessor = NODE_CLASS_MAPPINGS["DensePosePreprocessor"]()
        idm_vton = NODE_CLASS_MAPPINGS["IDM-VTON"]()
        saveimage = save_image_wrapper(ctx, SaveImage)()
        for q in range(args.queue_size):
            groundingdinosamsegment_segment_anything_6 = (
                groundingdinosamsegment_segment_anything.main(
                    prompt=args.gd_prompt,
                    threshold=0.3,
                    sam_model=get_value_at_index(sammodelloader_segment_anything_4, 0),
                    grounding_dino_model=get_value_at_index(
                        groundingdinomodelloader_segment_anything_5, 0
                    ),
                    image=get_value_at_index(loadimage_1, 0),
                )
            )

            masktoimage_8 = masktoimage.mask_to_image(
                mask=get_value_at_index(groundingdinosamsegment_segment_anything_6, 1)
            )

            denseposepreprocessor_12 = denseposepreprocessor.execute(
                model="densepose_r50_fpn_dl.torchscript",
                cmap="Viridis (MagicAnimate)",
                resolution=512,
                image=get_value_at_index(loadimage_1, 0),
            )

            idm_vton_11 = idm_vton.make_inference(
                garment_description=garment_desc,
                negative_prompt=negative_prompt,
                width=768,
                height=1024,
                num_inference_steps=30,
                guidance_scale=2,
                strength=0.7,
                seed=random.randint(1, 2**64),
                pipeline=get_value_at_index(pipelineloader_10, 0),
                human_img=get_value_at_index(loadimage_1, 0),
                pose_img=get_value_at_index(denseposepreprocessor_12, 0),
                mask_img=get_value_at_index(masktoimage_8, 0),
                garment_img=get_value_at_index(loadimage_2, 0),
            )

            if __name__ != "__main__":
                return dict(
                    filename_prefix="ComfyUI",
                    images=get_value_at_index(idm_vton_11, 0),
                    #prompt=PROMPT_DATA,
                )
            else:
                saveimage_30 = saveimage.save_images(
                    filename_prefix="ComfyUI",
                    images=get_value_at_index(idm_vton_11, 0),
                    #prompt=PROMPT_DATA,
                )


if __name__ == "__main__":
    main()
