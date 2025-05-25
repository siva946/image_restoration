from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image
import io
import base64
import torch
from diffusers import StableDiffusionInpaintPipeline
from gfpgan import GFPGANer
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"

inpaint_pipe = None
try:
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    inpaint_pipe = inpaint_pipe.to(device)
    if device == "cuda":
        inpaint_pipe.enable_model_cpu_offload()
        inpaint_pipe.unet = inpaint_pipe.unet.half()
        inpaint_pipe.vae = inpaint_pipe.vae.half()
        try:
            inpaint_pipe.unet = torch.compile(inpaint_pipe.unet)
            inpaint_pipe.vae = torch.compile(inpaint_pipe.vae)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
except Exception as e:
    print(f"Error loading inpaint pipeline: {e}")

gfpgan= GFPGANer(model_path='weights/GFPGANv1.4.pth',upscale=1,arch='clean',channel_multiplier=2
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        denoise = request.form.get('denoise') == 'true'
        denoise_strength = int(request.form.get('denoiseStrength', 10))
        upscale = request.form.get('upscale') == 'true'
        upscale_factor = int(request.form.get('upscaleFactor', 4))
        enhance = request.form.get('enhance') == 'true'
        enhance_intensity = int(request.form.get('enhanceIntensity', 3))
        face_enhance = request.form.get('faceEnhance') == 'true'
        face_enhance_scale = float(request.form.get('faceEnhanceScale', 1.0))
        prompt = request.form.get('prompt', '')
        guidance_scale = float(request.form.get('guidanceScale', 8.0))
        num_inference_steps = int(request.form.get('numInferenceSteps', 10))
        mask_file = request.files.get('mask', None)

        # Read image
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]

        # Denoising
        if denoise:
            image = cv2.fastNlMeansDenoisingColored(image, None, h=denoise_strength, hColor=denoise_strength, templateWindowSize=7, searchWindowSize=21)

        # Generative fill
        if mask_file and prompt and inpaint_pipe is not None:
            mask = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            # Resize image and mask to 512x512 for inpainting
            image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            if np.sum(mask_resized) > 0:
                image_pil = Image.fromarray(image_resized)
                mask_pil = Image.fromarray(mask_resized)
                inpaint_result = inpaint_pipe(
                    prompt=prompt,
                    image=image_pil,
                    mask_image=mask_pil,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                image_resized = np.array(inpaint_result.images[0])
            # Resize back to original size
            image = cv2.resize(image_resized, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)

        # Enhancement
        if enhance:
            clahe = cv2.createCLAHE(clipLimit=enhance_intensity, tileGridSize=(8, 8))
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            # Adjust sharpening kernel strength based on enhance_intensity
            kernel_strength = 1 + (enhance_intensity / 10)
            kernel = np.array([[0, -1 * kernel_strength, 0], [-1 * kernel_strength, 5 * kernel_strength, -1 * kernel_strength], [0, -1 * kernel_strength, 0]], dtype=np.float32)
            image = cv2.filter2D(image, -1, kernel)

        # Face enhancement
        if face_enhance:
            _, _, image = gfpgan.enhance(image, has_aligned=False, only_center_face=False, paste_back=True, upscale=face_enhance_scale)

        # Resize to original size or upscaled size
        if upscale:
            # Optional: integrate Real-ESRGAN for better upscaling if available
            target_shape = (original_shape[1] * upscale_factor, original_shape[0] * upscale_factor)  # scaled original size
            image = cv2.resize(image, target_shape, interpolation=cv2.INTER_LANCZOS4)
        else:
            target_shape = (original_shape[1], original_shape[0])  # Original size
            image = cv2.resize(image, target_shape, interpolation=cv2.INTER_AREA)

        # Encode result
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        restored_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'restored_image': restored_image})
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return detailed error message for debugging (can be changed to generic in production)
        return jsonify({'error': f'An error occurred during image processing: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
