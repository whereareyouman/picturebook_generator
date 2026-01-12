#!/usr/bin/env python3
"""
Enhanced Treasure Story Generator with Image Generation

This script generates customizable stories with AI-generated images using Google's Gemini AI.
Now supports dotenv for API key management and enhanced customization options.

Author: Assistant
Date: 2025-06-07
Version: 2.0 - Fixed API issues and improved error handling
"""

import json
import os
import base64
import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# WeasyPrint is optional. On macOS it's common to fail with missing system libraries
# (e.g. libgobject). We should not block the whole app when PDF deps are missing.
WEASYPRINT_AVAILABLE = False
WEASYPRINT_IMPORT_ERROR = None
try:
    from weasyprint import CSS, HTML  # noqa: F401
    WEASYPRINT_AVAILABLE = True
except Exception as e:  # ImportError / OSError / etc.
    WEASYPRINT_AVAILABLE = False
    WEASYPRINT_IMPORT_ERROR = str(e)

_TEXT_LENGTH_ALLOWED = {"one_sentence", "short", "standard", "long"}


def _normalize_text_length(value: str | None) -> str:
    if not value:
        return "standard"
    v = str(value).strip().lower()
    return v if v in _TEXT_LENGTH_ALLOWED else "standard"


def _text_length_requirement_line(text_length: str) -> str:
    """Bullet line used in Gemini prompt."""
    tl = _normalize_text_length(text_length)
    if tl == "one_sentence":
        return "- Keep each scene text to exactly ONE short sentence (caption-like). Aim for <= 25 Chinese characters or <= 20 English words."
    if tl == "short":
        return "- Keep each scene description to 1-2 short sentences."
    if tl == "long":
        return "- Keep each scene description to 4-6 sentences."
    return "- Keep each scene description between 2-4 sentences."


def _azure_text_length_rule(text_length: str) -> str:
    """Rule line used in Azure JSON plan prompt."""
    tl = _normalize_text_length(text_length)
    if tl == "one_sentence":
        return "- Each text is exactly 1 short sentence (caption-like)."
    if tl == "short":
        return "- Each text is 1-2 sentences, family-friendly."
    if tl == "long":
        return "- Each text is 4-6 sentences, family-friendly."
    return "- Each text is 2-4 sentences, family-friendly."


def _shorten_scene_text(text: str, text_length: str) -> str:
    """
    Best-effort post-processing to enforce short narration.
    This is a safety net when the model doesn't strictly follow instructions.
    """
    tl = _normalize_text_length(text_length)
    t = (text or "").strip()
    if not t:
        return t

    # Only post-process when user explicitly asked for shorter text.
    # Preserve original formatting (e.g., newlines) for standard/long.
    if tl not in {"one_sentence", "short"}:
        return t

    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(scene|Scene|SCENE)\s*\d+\s*[:\-–—]\s*", "", t).strip()
    t = re.sub(r"^(场景|镜头|章节)\s*\d+\s*[：:，,\-–—]\s*", "", t).strip()

    if tl == "one_sentence":
        m = re.search(r"[。！？.!?]", t)
        if m:
            t = t[: m.end()].strip()
        if len(t) > 120:
            t = t[:120].rstrip()
        return t

    if tl == "short":
        parts = re.split(r"(?<=[。！？.!?])\s+", t)
        t2 = " ".join(p for p in parts[:2] if p).strip()
        return t2 or t

    return t


def _get_ai_provider() -> str:
    """Return active AI provider: 'gemini' (default) or 'azure'."""
    provider = os.getenv("AI_PROVIDER", "gemini").strip().lower()
    if provider in {"azure", "azure_openai", "azure-openai"}:
        return "azure"
    return "gemini"


def _normalize_azure_api_version(api_version: str | None) -> str:
    """
    Normalize Azure OpenAI api_version.
    Accepts:
    - '2025-02-01-preview'
    - 'azure-openai-2025-02-01-preview' (tutorial-style)
    """
    if not api_version:
        return "2025-02-01-preview"
    v = api_version.strip()
    prefix = "azure-openai-"
    if v.startswith(prefix):
        v = v[len(prefix) :]
    return v


def _get_azure_openai_config() -> dict:
    """Read Azure OpenAI config from env/.env (dotenv is loaded in setup_client)."""
    endpoint = (
        os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
    )
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_version = _normalize_azure_api_version(
        os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
    )
    chat_deployment = (
        os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_MODEL")
        or "gpt-4o"
    )
    image_deployment = (
        os.getenv("AZURE_OPENAI_IMAGE_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DALLE_DEPLOYMENT")
        or ""
    )

    return {
        "endpoint": endpoint,
        "api_key": api_key,
        "api_version": api_version,
        "chat_deployment": chat_deployment,
        "image_deployment": image_deployment,
    }


def _setup_azure_openai_client():
    """Create AzureOpenAI client. Returns None if not configured."""
    cfg = _get_azure_openai_config()

    if not cfg["endpoint"] or not cfg["api_key"]:
        print("⚠️  Azure OpenAI 未配置（缺少 endpoint 或 api key）。")
        print("   请在 .env 中设置：AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY")
        return None

    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception as e:
        print("❌ 未安装 openai SDK，无法使用 Azure OpenAI。")
        print("   请先安装：pip install openai")
        print(f"   Import error: {e}")
        return None

    try:
        client = AzureOpenAI(
            api_key=cfg["api_key"],
            api_version=cfg["api_version"],
            azure_endpoint=cfg["endpoint"],
        )
        print("✅ Successfully configured Azure OpenAI client")
        print(f"   endpoint: {cfg['endpoint']}")
        print(f"   api_version: {cfg['api_version']}")
        print(f"   chat_deployment: {cfg['chat_deployment']}")
        if cfg["image_deployment"]:
            print(f"   image_deployment: {cfg['image_deployment']}")
        else:
            print("   image_deployment: (not set) -> will use placeholder images")
        return client
    except Exception as e:
        print(f"❌ Failed to initialize Azure OpenAI client: {e}")
        return None


def setup_client():
    """
    Initialize AI client based on AI_PROVIDER.

    - AI_PROVIDER=gemini (default): Google Gemini via google-genai
    - AI_PROVIDER=azure: Azure OpenAI via openai.AzureOpenAI

    Returns:
        Client instance or None if not configured

    Raises:
        ValueError: If API key is not found
    """
    # Load environment variables from .env file
    load_dotenv()

    provider = _get_ai_provider()
    if provider == "azure":
        return _setup_azure_openai_client()

    # Try to get API key from environment variable
    api_key = os.getenv('GOOGLE_API_KEY')

    if not api_key or api_key == 'your_google_api_key_here':
        print("⚠️  Google API key not found or not configured properly.")
        print("Please update the .env file with your actual API key.")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        api_key = input("Or enter your Google API key now: ").strip()

    if not api_key:
        raise ValueError("API key is required to use the image generation service")

    try:
        client = genai.Client(api_key=api_key)
        print("✅ Successfully connected to Google Gemini API")
        return client
    except Exception as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
        raise


def _extract_json_from_text(text: str) -> str:
    """Best-effort extraction of a JSON object string from model output."""
    if not text:
        return ""
    s = text.strip()
    # Remove fenced code blocks if present
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    s = s.strip()

    if s.startswith("{") and s.endswith("}"):
        return s

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def _create_placeholder_image(image_path: Path, scene_num: int, caption: str = "") -> tuple[int, int]:
    """Create a simple placeholder PNG so the UI/PDF pipeline can still work."""
    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    title = f"Scene {scene_num}"
    draw.text((40, 40), title, fill="black", font=font)
    if caption:
        # Keep caption short to avoid overflow
        caption_short = caption.strip().replace("\n", " ")
        if len(caption_short) > 200:
            caption_short = caption_short[:200] + "..."
        draw.text((40, 90), caption_short, fill="gray", font=font)

    img.save(image_path, "PNG")
    return img.size


def _azure_chat_generate_story_plan(
    client, story_prompt: str, num_scenes: int, text_length: str = "standard"
) -> dict:
    """Use Azure OpenAI chat model to generate a JSON scene plan."""
    cfg = _get_azure_openai_config()
    deployment = cfg["chat_deployment"]
    text_length = _normalize_text_length(text_length)
    text_rule = _azure_text_length_rule(text_length)

    system = (
        "You are a picture-book author. Return ONLY valid JSON (no markdown).\n"
        "Schema:\n"
        "{\n"
        '  \"title\": string,\n'
        '  \"scenes\": [\n'
        '    {\"scene_number\": 1, \"text\": string, \"image_prompt\": string},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        f"- EXACTLY {num_scenes} scenes.\n"
        f"{text_rule}\n"
        "- image_prompt describes ONE illustration, include art style, composition, characters.\n"
        "- Prefer: text in Chinese if user prompt is Chinese; image_prompt in English.\n"
    )
    user = (
        f'Story idea: \"{story_prompt}\"\n'
        f"Number of scenes: {num_scenes}\n"
        "Generate the JSON now."
    )

    resp = client.chat.completions.create(
        model=deployment,  # Azure: this is the deployment name
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.8,
        max_tokens=4096,
    )

    content = resp.choices[0].message.content if resp and resp.choices else ""
    json_str = _extract_json_from_text(content or "")
    data = json.loads(json_str)
    return data


def _azure_generate_image_bytes(client, prompt: str) -> bytes | None:
    """Use Azure OpenAI image deployment to generate an image. Returns bytes or None."""
    cfg = _get_azure_openai_config()
    deployment = cfg["image_deployment"]
    if not deployment:
        return None

    # Prefer base64 response to avoid extra downloads
    image_resp = client.images.generate(
        model=deployment,  # Azure: deployment name
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )

    if not image_resp or not getattr(image_resp, "data", None):
        return None

    item = image_resp.data[0]
    b64_json = getattr(item, "b64_json", None)
    if b64_json:
        return base64.b64decode(b64_json)

    url = getattr(item, "url", None)
    if url:
        # Fallback: download image bytes (requires network)
        import urllib.request

        with urllib.request.urlopen(url) as resp:
            return resp.read()

    return None


def _generate_custom_story_with_images_azure(
    client,
    story_prompt: str,
    num_scenes: int,
    output_dir: Path,
    delay_between_requests: int = 0,
    progress_callback=None,
    text_length: str = "standard",
):
    """Azure OpenAI flow: chat -> JSON scenes -> (optional) images per scene."""
    if progress_callback:
        progress_callback(0, num_scenes, "Generating story outline (Azure OpenAI)...")

    try:
        plan = _azure_chat_generate_story_plan(client, story_prompt, num_scenes, text_length=text_length)
    except Exception as e:
        print(f"❌ Azure chat generation failed: {e}")
        return None

    scenes = plan.get("scenes") if isinstance(plan, dict) else None
    if not isinstance(scenes, list) or not scenes:
        print("❌ Azure chat returned no scenes.")
        return None

    # Normalize and sort scenes by scene_number
    norm_scenes = []
    for s in scenes:
        if not isinstance(s, dict):
            continue
        sn = s.get("scene_number")
        try:
            sn_int = int(sn)
        except Exception:
            continue
        norm_scenes.append(
            {
                "scene_number": sn_int,
                "text": str(s.get("text", "")).strip(),
                "image_prompt": str(s.get("image_prompt", "")).strip(),
            }
        )

    norm_scenes = sorted(norm_scenes, key=lambda x: x["scene_number"])
    norm_scenes = [s for s in norm_scenes if 1 <= s["scene_number"] <= num_scenes]
    if len(norm_scenes) < num_scenes:
        print(f"⚠️  Azure returned {len(norm_scenes)} scenes, expected {num_scenes}. Will proceed.")

    cfg = _get_azure_openai_config()
    model_label = f"azure:{cfg['chat_deployment']}"
    if cfg["image_deployment"]:
        model_label += f"+images:{cfg['image_deployment']}"
    else:
        model_label += "+images:placeholder"

    story_data = {
        "scenes": [],
        "generated_at": datetime.now().isoformat(),
        "model": model_label,
        "original_prompt": story_prompt,
        "num_scenes": num_scenes,
        "text_length": _normalize_text_length(text_length),
    }

    # Generate/save images
    for idx, scene in enumerate(norm_scenes, start=1):
        scene_num = scene["scene_number"]
        scene_text = _shorten_scene_text(scene["text"] or "", text_length)
        image_prompt = scene["image_prompt"] or ""

        story_data["scenes"].append(
            {
                "type": "text",
                "content": scene_text,
                "scene_number": scene_num,
                "part_index": (scene_num - 1) * 2,
            }
        )

        if progress_callback:
            progress_callback(scene_num - 1, num_scenes, f"Generating image for scene {scene_num}...")

        image_filename = f"scene_{scene_num:02d}.png"
        image_path = output_dir / image_filename

        img_bytes = None
        if image_prompt:
            try:
                img_bytes = _azure_generate_image_bytes(client, image_prompt)
            except Exception as e:
                print(f"⚠️  Azure image generation failed (scene {scene_num}): {e}")
                img_bytes = None

        if img_bytes:
            try:
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                img.save(image_path, "PNG")
                image_size = img.size
            except Exception as e:
                print(f"⚠️  Failed to decode/save image bytes (scene {scene_num}): {e}")
                image_size = _create_placeholder_image(image_path, scene_num, image_prompt or scene_text)
        else:
            image_size = _create_placeholder_image(image_path, scene_num, image_prompt or scene_text)

        story_data["scenes"].append(
            {
                "type": "image",
                "filename": image_filename,
                "path": str(image_path),
                "scene_number": scene_num,
                "part_index": (scene_num - 1) * 2 + 1,
                "image_size": image_size,
            }
        )

        if delay_between_requests and idx < len(norm_scenes):
            time.sleep(delay_between_requests)

    return story_data


def generate_custom_story_with_images(
    client,
    story_prompt,
    num_scenes,
    output_dir,
    delay_between_requests=6,
    progress_callback=None,
    text_length: str = "standard",
):
    """
    Generate a custom story with images based on user input.

    Args:
        client (genai.Client): Configured GenAI client
        story_prompt (str): User-defined story prompt
        num_scenes (int): Number of scenes to generate
        output_dir (Path): Directory to save images
        delay_between_requests (int): Seconds to wait between requests (rate limiting)

    Returns:
        dict: Story data with text and image paths
    """
    provider = _get_ai_provider()
    if provider == "azure":
        return _generate_custom_story_with_images_azure(
            client,
            story_prompt,
            num_scenes,
            Path(output_dir),
            delay_between_requests=delay_between_requests,
            progress_callback=progress_callback,
            text_length=text_length,
        )

    # Use the image generation model (Gemini multimodal)
    model = "gemini-2.5-flash-image"

    # Enhanced prompt for better story generation
    text_length_line = _text_length_requirement_line(text_length)
    full_prompt = f"""
    You are an expert storyteller and illustrator creating a captivating picture book.

    Create a {num_scenes}-scene story based on this idea: "{story_prompt}"

    Requirements:
    - Each scene should advance the story and be distinct
    - Include vivid, engaging descriptions suitable for illustration
    - Make it artistic, engaging, and age-appropriate
    - Generate both descriptive text and a corresponding image for each scene
    {text_length_line}

    Please create exactly {num_scenes} scenes, each with descriptive text and an accompanying image.
    Structure: Scene 1: [description], Scene 2: [description], etc.
    """

    print(f"🎨 Generating custom story: '{story_prompt}'")
    print(f"📊 Scenes to generate: {num_scenes}")
    print("⏳ This may take several minutes due to rate limiting...")
    print("⏱️  Rate limit: ~6 seconds between requests (10/minute limit)")

    try:
        print("🔄 Making API request...")

        # Create the configuration properly
        config = types.GenerateContentConfig(
            response_modalities=["Text", "Image"],
            max_output_tokens=8192
        )

        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=config
        )

        # Debug: Print response structure
        print("🔍 API Response received, processing...")

        if not response:
            raise ValueError("No response received from API")

        if not hasattr(response, 'candidates') or not response.candidates:
            raise ValueError("No candidates in API response")

        if not response.candidates[0]:
            raise ValueError("First candidate is empty")

        if not hasattr(response.candidates[0], 'content') or not response.candidates[0].content:
            raise ValueError("No content in first candidate")

        if not hasattr(response.candidates[0].content, 'parts') or not response.candidates[0].content.parts:
            raise ValueError("No parts in content")

        story_data = {
            'scenes': [],
            'generated_at': datetime.now().isoformat(),
            'model': model,
            'original_prompt': story_prompt,
            'num_scenes': num_scenes,
            'text_length': _normalize_text_length(text_length),
            'total_parts': len(response.candidates[0].content.parts)
        }

        scene_counter = 1
        total_parts = len(response.candidates[0].content.parts)

        print(f"📦 Processing {total_parts} parts from API response...")

        for i, part in enumerate(response.candidates[0].content.parts):
            print(f"🔄 Processing part {i+1}/{total_parts}...")

            if hasattr(part, 'text') and part.text is not None:
                print(f"📖 Found text content (Scene {scene_counter})")
                MAX_TEXT_PREVIEW_LENGTH = 200
                text_preview = part.text[:MAX_TEXT_PREVIEW_LENGTH] + "..." if len(part.text) > MAX_TEXT_PREVIEW_LENGTH else part.text
                print(f"   Preview: {text_preview}")
                scene_text = _shorten_scene_text(part.text, text_length)

                story_data['scenes'].append({
                    'type': 'text',
                    'content': scene_text,
                    'scene_number': scene_counter if scene_counter <= num_scenes else 'additional',
                    'part_index': i
                })

            elif hasattr(part, 'inline_data') and part.inline_data is not None:
                print(f"🖼️  Found image data (Scene {scene_counter})")

                try:
                    # Save image to file
                    image = Image.open(BytesIO(part.inline_data.data))
                    image_filename = f"scene_{scene_counter:02d}.png"
                    image_path = output_dir / image_filename

                    # Save image with error handling
                    image.save(image_path, 'PNG')
                    print(f"✅ Scene {scene_counter} image saved: {image_filename}")

                    story_data['scenes'].append({
                        'type': 'image',
                        'filename': image_filename,
                        'path': str(image_path),
                        'scene_number': scene_counter,
                        'part_index': i,
                        'image_size': image.size
                    })

                    scene_counter += 1

                    # Rate limiting delay (respect 10 requests per minute)
                    if scene_counter <= num_scenes and i < total_parts - 1:
                        print(f"⏳ Waiting {delay_between_requests} seconds (rate limiting)...")
                        time.sleep(delay_between_requests)

                except Exception as img_error:
                    print(f"❌ Error saving image {scene_counter}: {img_error}")
                    continue
            else:
                print(f"⚠️  Unknown part type at index {i}")

        # Save story metadata
        metadata_path = output_dir / "story_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(story_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Generated {scene_counter-1} scene images")
        print(f"📊 Total parts processed: {total_parts}")
        print(f"💾 Metadata saved: {metadata_path}")

        return story_data

    except Exception as e:
        print(f"❌ Error generating story: {e}")
        print(f"🔍 Error type: {type(e).__name__}")

        if "quota" in str(e).lower() or "rate" in str(e).lower():
            print("💡 This might be due to rate limiting. Try again later or reduce the number of scenes.")
        elif "401" in str(e) or "unauthorized" in str(e).lower():
            print("💡 Check your API key - it might be invalid or expired.")
        elif "model" in str(e).lower():
            print("💡 The model might not be available. Try using 'gemini-2.5-flash' instead.")

        return None


def create_html_display(story_data, output_dir):
    """
    Create an HTML file to display the custom story with images.

    Args:
        story_data (dict): Story data with text and images
        output_dir (Path): Directory containing images

    Returns:
        str: Path to HTML file
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Custom AI Story - {story_data.get('original_prompt', 'Adventure')}</title>
    <style>
        body {{
            font-family: 'Comic Sans MS', cursive, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #f0f8ff, #e6e6fa, #f5deb3);
            min-height: 100vh;
        }}
        .header {{
            text-align: center;
            color: #4a4a4a;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 15px;
            border: 3px solid #daa520;
        }}
        .story-info {{
            background: rgba(135, 206, 235, 0.9);
            border: 2px solid #4169e1;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            color: #2f4f4f;
        }}
        .scene {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            margin: 20px 0;
            padding: 20px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            border: 3px solid #daa520;
        }}
        .scene-image {{
            width: 100%;
            max-width: 600px;
            height: auto;
            border-radius: 10px;
            border: 2px solid #8b4513;
            margin: 15px 0;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}
        .scene-text {{
            font-size: 16px;
            line-height: 1.6;
            color: #2f4f4f;
            text-align: justify;
            margin: 10px 0;
        }}
        .scene-number {{
            color: #b8860b;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .generated-info {{
            text-align: center;
            font-size: 12px;
            color: #696969;
            margin-top: 30px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
        }}
        .debug-info {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Custom AI Story 🎨</h1>
        <h2>"{story_data.get('original_prompt', 'Adventure Story')}"</h2>
    </div>

    <div class="story-info">
        <h3>📖 Story Details:</h3>
        <ul>
            <li><strong>Original Prompt:</strong> {story_data.get('original_prompt', 'N/A')}</li>
            <li><strong>Scenes Requested:</strong> {story_data.get('num_scenes', 'N/A')}</li>
            <li><strong>Total Parts Generated:</strong> {story_data.get('total_parts', 'N/A')}</li>
            <li><strong>Generated:</strong> {story_data.get('generated_at', 'N/A')}</li>
            <li><strong>Model:</strong> {story_data.get('model', 'N/A')}</li>
        </ul>
    </div>
"""

    # Group scenes by number for proper ordering
    text_scenes = {}
    image_scenes = {}

    for scene in story_data['scenes']:
        scene_num = scene['scene_number']
        if scene['type'] == 'text':
            if scene_num not in text_scenes:
                text_scenes[scene_num] = []
            text_scenes[scene_num].append(scene['content'])
        elif scene['type'] == 'image':
            image_scenes[scene_num] = scene

    # Generate HTML for each scene
    scene_numbers = sorted(set(list(text_scenes.keys()) + list(image_scenes.keys())))
    actual_scenes = [sn for sn in scene_numbers if isinstance(sn, int)]

    for scene_num in actual_scenes:
        html_content += '    <div class="scene">\n'
        html_content += f'        <div class="scene-number">Scene {scene_num}</div>\n'

        if scene_num in image_scenes:
            image_scene = image_scenes[scene_num]
            html_content += f'        <img src="{image_scene["filename"]}" alt="Scene {scene_num}" class="scene-image">\n'
            if 'image_size' in image_scene:
                html_content += f'        <div class="debug-info">Image size: {image_scene["image_size"]}</div>\n'

        if scene_num in text_scenes:
            for text_content in text_scenes[scene_num]:
                # Split text into paragraphs for better formatting
                paragraphs = text_content.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        # Simple markdown-style processing
                        processed_paragraph = paragraph.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
                        html_content += f'        <div class="scene-text">{processed_paragraph.strip()}</div>\n'

        html_content += '    </div>\n\n'

    # Add any additional text content
    for scene in story_data['scenes']:
        if scene['type'] == 'text' and scene['scene_number'] == 'additional':
            html_content += '    <div class="scene">\n'
            html_content += '        <div class="scene-number">Additional Content</div>\n'
            html_content += f'        <div class="scene-text">{scene["content"]}</div>\n'
            html_content += '    </div>\n\n'

    html_content += f"""
    <div class="generated-info">
        <p>Generated on: {story_data['generated_at']}</p>
        <p>Model: {story_data['model']}</p>
        <p>Total Parts: {story_data.get('total_parts', 'N/A')}</p>
        <p>✨ Created with Google Gemini AI ✨</p>
    </div>
</body>
</html>
"""

    # Create safe filename
    safe_prompt = "".join(c for c in story_data.get('original_prompt', 'story')[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    html_filename = f"{safe_prompt.replace(' ', '_')}_story.html"
    html_path = output_dir / html_filename

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return str(html_path)


def create_pdf_from_html(html_path, output_dir):
    """
    Convert HTML story to PDF for easy sharing.
    Now uses enhanced PDF generation with proper page breaks.

    Args:
        html_path (str): Path to the HTML file
        output_dir (Path): Directory to save PDF

    Returns:
        str: Path to PDF file or None if failed
    """
    if not WEASYPRINT_AVAILABLE:
        print("⚠️  WeasyPrint not available. PDF generation skipped.")
        if WEASYPRINT_IMPORT_ERROR:
            print(f"   Import error: {WEASYPRINT_IMPORT_ERROR}")
        print("   Install with: uv pip install weasyprint")
        print("   macOS 还需要系统依赖 / macOS system deps:")
        print("   brew install pango cairo gdk-pixbuf libffi glib")
        return None

    # Import here to avoid unbound variable error
    from weasyprint import CSS, HTML

    try:
        html_file = Path(html_path)
        if not html_file.exists():
            print(f"❌ HTML file not found: {html_path}")
            return None

        # Create PDF filename
        pdf_filename = html_file.stem + '.pdf'
        pdf_path = output_dir / pdf_filename

        print(f"📄 Converting to PDF with enhanced formatting: {pdf_filename}")

        # Convert HTML to PDF with improved settings
        # Enhanced CSS for better PDF formatting
        enhanced_css = CSS(string="""
            @page {
                size: A4;
                margin: 20mm;
            }
            .scene {
                page-break-before: always;
                page-break-inside: avoid;
            }
            .scene-image {
                max-width: 100%;
                max-height: 15cm;
                page-break-inside: avoid;
            }
            body {
                font-family: 'Times New Roman', serif;
                font-size: 12pt;
                line-height: 1.4;
            }
            .header {
                page-break-after: always;
            }
            .story-info {
                page-break-after: always;
            }
            .debug-info {
                display: none;
            }
        """)

        html_doc = HTML(filename=str(html_file))
        html_doc.write_pdf(str(pdf_path), stylesheets=[enhanced_css])
        print(f"✅ Enhanced PDF created: {pdf_path}")
        print("📖 Each scene now starts on a new page for better readability")
        return str(pdf_path)

    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        print("💡 This might be due to missing system dependencies.")
        print("   On Ubuntu/Debian: sudo apt-get install libpango-1.0-0 libharfbuzz0b libcairo-gobject2")
        return None


def test_api_connection():
    """Test API connection and available models."""
    try:
        provider = _get_ai_provider()
        client = setup_client()
        if not client:
            return False

        if provider == "azure":
            cfg = _get_azure_openai_config()
            dep = cfg["chat_deployment"]
            resp = client.chat.completions.create(
                model=dep,
                messages=[{"role": "user", "content": "Reply with 'API test successful'."}],
                max_tokens=50,
            )
            text = resp.choices[0].message.content if resp and resp.choices else ""
            if text:
                print("✅ Azure OpenAI connection test successful!")
                print(f"🤖 Response: {text[:100]}...")
                return True
            print("❌ Azure OpenAI test failed - no response text")
            return False

        # Gemini (default)
        test_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello and confirm you're working."
        )

        if test_response and test_response.candidates:
            candidate = test_response.candidates[0]
            if candidate and candidate.content and candidate.content.parts and candidate.content.parts[0].text:
                print("✅ API connection test successful!")
                print(f"🤖 Response: {candidate.content.parts[0].text[:100]}...")
                return True
            else:
                print("❌ API test failed - response structure unexpected or missing text.")
                return False
        else:
            print("❌ API test failed - no response or no candidates")
            return False

    except Exception as e:
        print(f"❌ API connection test failed: {e}")
        return False


def main():
    """Main function to orchestrate the custom story generation process."""
    print("🎨 Welcome to the Enhanced Custom Story Generator! 🎨")
    print("=" * 70)

    # Setup output directory
    project_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_dir / "generated_stories" / f"story_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Test API connection first
        print("\n🔍 Testing API connection...")
        if not test_api_connection():
            print("❌ API connection failed. Please check your API key and try again.")
            return None

        # Initialize client
        client = setup_client()

        # Get user input for custom story
        print("\n📝 Let's create your custom story!")
        story_prompt = input("What story would you like to create? (e.g., 'A robot learning to paint'): ").strip()

        if not story_prompt:
            story_prompt = "A brave explorer discovering magical creatures in an enchanted forest"
            print(f"Using default story: {story_prompt}")

        num_scenes_input = input("How many scenes? (1-9999+, your choice!): ").strip()
        try:
            num_scenes = int(num_scenes_input)
            if num_scenes < 1:
                num_scenes = 1
                print("Setting minimum: 1 scene")
            # No upper limit - user's choice!
        except ValueError:
            num_scenes = 6
            print("Using default: 6 scenes")

        # Show rate limiting info
        estimated_time = num_scenes * 6 / 60  # 6 seconds per request
        hours = int(estimated_time // 60)
        minutes = int(estimated_time % 60)

        if hours > 0:
            time_str = f"~{hours}h {minutes}m"
        else:
            time_str = f"~{estimated_time:.1f} minutes"

        print(f"\n⏱️  Estimated generation time: {time_str}")
        print("💡 This is due to API rate limits (10 requests/minute)")

        LARGE_STORY_THRESHOLD = 100
        LONG_STORY_THRESHOLD = 50
        if num_scenes > LARGE_STORY_THRESHOLD:
            print(f"🎯 Large story! {num_scenes} scenes will create an epic adventure!")
        elif num_scenes > LONG_STORY_THRESHOLD:
            print(f"📚 Long story! {num_scenes} scenes will make a wonderful book!")

        proceed = input("Continue? (y/n): ").strip().lower()
        if proceed not in ['y', 'yes', '']:
            print("Story generation cancelled.")
            return None

        # Generate story with images
        story_data = generate_custom_story_with_images(client, story_prompt, num_scenes, output_dir)

        if story_data:
            print("\n✅ Story generation completed successfully!")

            # Create HTML display
            html_path = create_html_display(story_data, output_dir)
            print(f"📄 HTML story created: {html_path}")

            # Create PDF version
            pdf_path = create_pdf_from_html(html_path, output_dir)
            if pdf_path:
                print(f"📄 PDF story created: {pdf_path}")

            print(f"\n📂 All files saved in: {output_dir}")
            print("\n🎉 Your custom adventure is ready!")
            print(f"💻 Open {html_path} in your browser to view the story")
            if pdf_path:
                print(f"📱 Share the PDF: {pdf_path}")

            return story_data
        else:
            print("❌ Failed to generate story")
            return None

    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


if __name__ == "__main__":
    main()
