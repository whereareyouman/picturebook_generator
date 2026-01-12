#!/usr/bin/env python3
"""
Enhanced PDF Generator for AI Story Generator

This module provides improved PDF generation with proper page breaks,
print-optimized styling, and better content alignment.

Author: Assistant
Date: 2025-05-28
"""

import os
from pathlib import Path
from datetime import datetime
import textwrap

WEASYPRINT_AVAILABLE = False
WEASYPRINT_IMPORT_ERROR = None
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except Exception as e:  # ImportError / OSError / etc.
    HTML = None
    CSS = None
    WEASYPRINT_AVAILABLE = False
    WEASYPRINT_IMPORT_ERROR = str(e)

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_AVAILABLE = False


def _load_font(size: int):
    """Load a reasonable font across platforms; fallback to default."""
    if not PIL_AVAILABLE:
        return None

    # Common fonts on macOS and Linux
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",  # macOS Chinese
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]

    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            continue

    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _wrap_text_to_width(text: str, draw: "ImageDraw.ImageDraw", font: "ImageFont.ImageFont", max_width: int):
    """Wrap text into lines that fit max_width (pixel-based)."""
    if not text:
        return []

    # Normalize whitespace but keep explicit newlines as paragraph breaks
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines: list[str] = []

    for para in paragraphs:
        if not para:
            lines.append("")
            continue

        # If it contains spaces, wrap by words; otherwise wrap by characters (works for CJK)
        tokens = para.split(" ") if " " in para else list(para)

        current = ""
        for token in tokens:
            candidate = (current + (" " if current and " " in para else "") + token).strip() if " " in para else (current + token)
            bbox = draw.textbbox((0, 0), candidate, font=font)
            width = bbox[2] - bbox[0]
            if width <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = token if " " in para else token

        if current:
            lines.append(current)

    # Remove trailing empty lines
    while lines and lines[-1] == "":
        lines.pop()

    return lines


def create_simple_pdf_with_pillow(story_data: dict, output_dir: Path) -> str | None:
    """
    Fallback PDF generation without WeasyPrint.
    Creates a multi-page PDF by composing each scene into an A4-like canvas using Pillow.
    """
    if not PIL_AVAILABLE:
        print("❌ Pillow (PIL) not available. Cannot generate PDF without WeasyPrint.")
        return None

    try:
        title = story_data.get("original_prompt", "story")
        safe_prompt = "".join(
            c for c in str(title)[:30] if c.isalnum() or c in (" ", "-", "_")
        ).strip() or "story"
        pdf_filename = f"{safe_prompt.replace(' ', '_')}_pillow.pdf"
        pdf_path = output_dir / pdf_filename

        # A4-ish canvas at ~200 DPI
        page_w, page_h = 1654, 2339
        margin = 90

        title_font = _load_font(56)
        header_font = _load_font(44)
        body_font = _load_font(30)

        pages: list[Image.Image] = []

        # Build index for scenes
        images_by_scene: dict[int, Path] = {}
        texts_by_scene: dict[int, list[str]] = {}

        for scene in story_data.get("scenes", []):
            scene_num = scene.get("scene_number")
            if not isinstance(scene_num, int):
                continue

            if scene.get("type") == "image":
                p = scene.get("path")
                if p:
                    img_path = Path(p)
                else:
                    img_path = output_dir / scene.get("filename", "")
                if img_path.exists():
                    images_by_scene[scene_num] = img_path
            elif scene.get("type") == "text":
                content = scene.get("content")
                if content:
                    texts_by_scene.setdefault(scene_num, []).append(str(content).strip())

        num_scenes = story_data.get("num_scenes")
        if not isinstance(num_scenes, int) or num_scenes <= 0:
            if images_by_scene:
                num_scenes = max(images_by_scene.keys())
            elif texts_by_scene:
                num_scenes = max(texts_by_scene.keys())
            else:
                num_scenes = 0

        if num_scenes <= 0:
            print("❌ No scenes found. Cannot generate PDF.")
            return None

        # Cover page
        cover = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(cover)
        y = margin
        draw.text((margin, y), "AI Picture Book", fill="black", font=header_font)
        y += 90
        draw.text((margin, y), str(title), fill="black", font=title_font)
        y += 140
        meta = f"Scenes: {num_scenes}    Generated: {story_data.get('generated_at', '')[:19]}"
        draw.text((margin, y), meta, fill="black", font=body_font)
        pages.append(cover)

        # Scene pages
        for scene_num in range(1, num_scenes + 1):
            scene_text = "\n".join(texts_by_scene.get(scene_num, [])).strip()
            img_path = images_by_scene.get(scene_num)

            # Base page
            page = Image.new("RGB", (page_w, page_h), "white")
            draw = ImageDraw.Draw(page)

            # Header
            y = margin
            draw.text((margin, y), f"Scene {scene_num}", fill="black", font=header_font)
            y += 70

            # Image
            image_area_h = int(page_h * 0.58)
            if img_path and img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    max_w = page_w - margin * 2
                    max_h = image_area_h
                    img.thumbnail((max_w, max_h))
                    x = margin + (max_w - img.width) // 2
                    page.paste(img, (x, y))
                    y += img.height + 40
                except Exception as e:
                    draw.text((margin, y), f"[Image load failed: {e}]", fill="red", font=body_font)
                    y += 60
            else:
                draw.text((margin, y), "[No image found]", fill="gray", font=body_font)
                y += 60

            # Text
            max_text_w = page_w - margin * 2
            if scene_text:
                lines = _wrap_text_to_width(scene_text, draw, body_font, max_text_w)
            else:
                lines = []

            line_h = 40
            max_lines = max(1, (page_h - margin - y) // line_h)

            # Render lines; if overflow, create continuation pages
            remaining = lines
            first = True
            while True:
                current_page = page if first else Image.new("RGB", (page_w, page_h), "white")
                current_draw = draw if first else ImageDraw.Draw(current_page)

                if not first:
                    cy = margin
                    current_draw.text((margin, cy), f"Scene {scene_num} (cont.)", fill="black", font=header_font)
                    cy += 70
                else:
                    cy = y

                chunk = remaining[:max_lines]
                for ln in chunk:
                    current_draw.text((margin, cy), ln, fill="black", font=body_font)
                    cy += line_h

                pages.append(current_page)
                remaining = remaining[max_lines:]
                if not remaining:
                    break
                first = False

        # Save multi-page PDF
        pages_rgb = [p.convert("RGB") for p in pages]
        pages_rgb[0].save(
            pdf_path,
            "PDF",
            save_all=True,
            append_images=pages_rgb[1:],
            resolution=200.0,
        )
        print(f"✅ Pillow PDF created: {pdf_path}")
        return str(pdf_path)

    except Exception as e:
        print(f"❌ Pillow PDF generation failed: {e}")
        return None


def create_print_optimized_html(story_data, output_dir):
    """
    Create an HTML file optimized for PDF/print generation.

    Args:
        story_data (dict): Story data with text and images
        output_dir (Path): Directory containing images

    Returns:
        str: Path to print-optimized HTML file
    """

    # Print-optimized CSS
    print_css = """
        /* Print-optimized styles */
        @page {
            size: A4;
            margin: 20mm 15mm;
            counter-increment: page;

            @bottom-center {
                content: "Page " counter(page);
                font-size: 10pt;
                color: #666;
            }
        }

        body {
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.4;
            color: #000;
            margin: 0;
            padding: 0;
            background: white;
        }

        /* Header styling */
        .header {
            text-align: center;
            margin-bottom: 30pt;
            padding-bottom: 20pt;
            border-bottom: 2pt solid #333;
            page-break-after: always;
        }

        .header h1 {
            font-size: 24pt;
            font-weight: bold;
            color: #333;
            margin: 0 0 10pt 0;
            text-shadow: none;
        }

        .header h2 {
            font-size: 16pt;
            font-weight: normal;
            color: #666;
            margin: 0;
            font-style: italic;
        }

        /* Story info */
        .story-info {
            background: #f8f8f8;
            border: 1pt solid #ccc;
            border-radius: 0;
            padding: 15pt;
            margin: 0 0 20pt 0;
            page-break-inside: avoid;
            page-break-after: always;
        }

        .story-info h3 {
            font-size: 14pt;
            margin: 0 0 10pt 0;
            color: #333;
        }

        .story-info ul {
            margin: 0;
            padding-left: 20pt;
        }

        .story-info li {
            margin-bottom: 5pt;
            font-size: 11pt;
        }

        /* Scene styling - each scene on its own page */
        .scene {
            page-break-before: always;
            page-break-inside: avoid;
            margin: 0;
            padding: 0;
            min-height: 200mm; /* Ensure minimum height for proper page breaks */
        }

        .scene-number {
            font-size: 18pt;
            font-weight: bold;
            color: #333;
            margin: 0 0 20pt 0;
            text-align: center;
            border-bottom: 1pt solid #ccc;
            padding-bottom: 10pt;
        }

        /* Image styling for print */
        .scene-image {
            width: 100%;
            max-width: 170mm; /* A4 width minus margins */
            max-height: 120mm; /* Leave room for text */
            height: auto;
            display: block;
            margin: 0 auto 20pt auto;
            border: 1pt solid #ccc;
            page-break-inside: avoid;
        }

        /* Text styling */
        .scene-text {
            font-size: 12pt;
            line-height: 1.6;
            color: #000;
            text-align: justify;
            margin: 10pt 0;
            hyphens: auto;
            text-indent: 15pt;
        }

        /* Remove web-specific styling for print */
        .scene-text strong {
            font-weight: bold;
        }

        /* Generated info */
        .generated-info {
            text-align: center;
            font-size: 9pt;
            color: #666;
            margin-top: 30pt;
            padding: 10pt;
            border-top: 1pt solid #ccc;
            page-break-before: always;
        }

        /* Table of contents */
        .toc {
            page-break-after: always;
            margin-bottom: 30pt;
        }

        .toc h2 {
            font-size: 18pt;
            font-weight: bold;
            color: #333;
            margin-bottom: 20pt;
            text-align: center;
            border-bottom: 2pt solid #333;
            padding-bottom: 10pt;
        }

        .toc-item {
            margin: 8pt 0;
            font-size: 12pt;
            display: flex;
            justify-content: space-between;
            border-bottom: 1pt dotted #ccc;
            padding-bottom: 3pt;
        }

        .toc-title {
            flex-grow: 1;
            margin-right: 10pt;
        }

        .toc-page {
            font-weight: bold;
        }

        /* Prevent widows and orphans */
        p, .scene-text {
            orphans: 3;
            widows: 3;
        }

        /* Hide decorative elements in print */
        .box-shadow, .border-radius, .gradient {
            display: none;
        }
    """

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{story_data.get('original_prompt', 'AI Story')}</title>
    <style>
        {print_css}
    </style>
</head>
<body>
    <!-- Cover Page -->
    <div class="header">
        <h1>AI Generated Story</h1>
        <h2>"{story_data.get('original_prompt', 'Adventure Story')}"</h2>
    </div>

    <!-- Story Information -->
    <div class="story-info">
        <h3>Story Details:</h3>
        <ul>
            <li><strong>Title:</strong> {story_data.get('original_prompt', 'N/A')}</li>
            <li><strong>Scenes:</strong> {story_data.get('num_scenes', 'N/A')}</li>
            <li><strong>Generated:</strong> {story_data.get('generated_at', 'N/A')[:19].replace('T', ' ')}</li>
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

    # Debug: Print what we found
    print(f"🔍 Found {len(text_scenes)} text scenes and {len(image_scenes)} image scenes")
    if len(text_scenes) == 0:
        print("⚠️  No text scenes found - this will create a PDF with only images!")

    # Create table of contents
    scene_numbers = sorted(set(list(text_scenes.keys()) + list(image_scenes.keys())))
    valid_scenes = [num for num in scene_numbers if isinstance(num, int)]

    if len(valid_scenes) > 3:  # Only add TOC if we have several scenes
        html_content += """
    <!-- Table of Contents -->
    <div class="toc">
        <h2>Table of Contents</h2>
"""
        for i, scene_num in enumerate(valid_scenes):
            # Estimate page number (cover + info + toc + scenes)
            page_num = 3 + i + 1
            scene_title = f"Scene {scene_num}"
            # Try to get first few words of text for better TOC
            if scene_num in text_scenes and text_scenes[scene_num]:
                first_text = text_scenes[scene_num][0][:50].strip()
                if len(first_text) > 30:
                    first_text = first_text[:30] + "..."
                scene_title += f": {first_text}"

            html_content += f"""
        <div class="toc-item">
            <span class="toc-title">{scene_title}</span>
            <span class="toc-page">{page_num}</span>
        </div>"""

        html_content += """
    </div>
"""

    # Generate HTML for each scene (each on its own page)
    for scene_num in valid_scenes:
        html_content += f'''
    <div class="scene">
        <div class="scene-number">Scene {scene_num}</div>
'''

        # Add image first if available
        if scene_num in image_scenes:
            image_scene = image_scenes[scene_num]
            html_content += f'        <img src="{image_scene["filename"]}" alt="Scene {scene_num}" class="scene-image">\n'

        # Add text content
        if scene_num in text_scenes:
            for text_content in text_scenes[scene_num]:
                # Clean up text formatting
                paragraphs = text_content.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        # Remove any HTML tags for clean print
                        clean_paragraph = paragraph.strip()
                        # Remove markdown-style bold markers if present
                        clean_paragraph = clean_paragraph.replace('**', '')
                        # Remove scene labels if they exist
                        if clean_paragraph.startswith(f'Scene {scene_num}:'):
                            clean_paragraph = clean_paragraph[len(f'Scene {scene_num}:'):].strip()
                        elif clean_paragraph.startswith(f'**Scene {scene_num}:**'):
                            clean_paragraph = clean_paragraph[len(f'**Scene {scene_num}:**'):].strip()

                        html_content += f'        <div class="scene-text">{clean_paragraph}</div>\n'

        html_content += '    </div>\n\n'

    # Add any additional content
    additional_content = []
    for scene in story_data['scenes']:
        if scene['type'] == 'text' and scene['scene_number'] == 'additional':
            additional_content.append(scene['content'])

    if additional_content:
        html_content += '''
    <div class="scene">
        <div class="scene-number">Additional Content</div>
'''
        for content in additional_content:
            html_content += f'        <div class="scene-text">{content}</div>\n'
        html_content += '    </div>\n\n'

    # Footer
    html_content += f"""
    <div class="generated-info">
        <p>Generated on: {story_data['generated_at'][:19].replace('T', ' ')}</p>
        <p>Model: {story_data['model']}</p>
        <p>Created with Google Gemini AI</p>
    </div>
</body>
</html>
"""

    # Create filename for print version
    safe_prompt = "".join(c for c in story_data.get('original_prompt', 'story')[:30]
                         if c.isalnum() or c in (' ', '-', '_')).rstrip()
    html_filename = f"{safe_prompt.replace(' ', '_')}_print.html"
    html_path = output_dir / html_filename

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return str(html_path)


def create_enhanced_pdf(story_data, output_dir):
    """
    Create a properly formatted PDF with page breaks and print optimization.

    Args:
        story_data (dict): Story data with text and images
        output_dir (Path): Directory to save PDF

    Returns:
        str: Path to PDF file or None if failed
    """
    if not WEASYPRINT_AVAILABLE:
        print("⚠️  WeasyPrint not available. PDF generation skipped.")
        print("   Install with: pip install weasyprint")
        if WEASYPRINT_IMPORT_ERROR:
            print(f"   Import error: {WEASYPRINT_IMPORT_ERROR}")
        print("   macOS system deps (Homebrew):")
        print("   brew install pango cairo gdk-pixbuf libffi glib")
        print("➡️  Falling back to Pillow PDF (no system deps needed)...")
        return create_simple_pdf_with_pillow(story_data, output_dir)

    try:
        print("📄 Creating print-optimized HTML...")
        html_path = create_print_optimized_html(story_data, output_dir)

        if not Path(html_path).exists():
            print(f"❌ HTML file not found: {html_path}")
            return None

        # Create PDF filename
        safe_prompt = "".join(c for c in story_data.get('original_prompt', 'story')[:30]
                             if c.isalnum() or c in (' ', '-', '_')).rstrip()
        pdf_filename = f"{safe_prompt.replace(' ', '_')}_enhanced.pdf"
        pdf_path = output_dir / pdf_filename

        print(f"📄 Converting to enhanced PDF: {pdf_filename}")
        print("⏳ This may take a moment for proper formatting...")

        # Convert HTML to PDF with enhanced settings
        if HTML is None:
            print("❌ WeasyPrint HTML class is not available. Cannot generate PDF.")
            return None
        html_doc = HTML(filename=html_path)

        # Write PDF with specific settings for better output
        html_doc.write_pdf(
            str(pdf_path),
            optimize_images=True,  # Optimize images for smaller file size
            jpeg_quality=85,       # Good quality/size balance
            pdf_version='1.7'      # Modern PDF version
        )

        print(f"✅ Enhanced PDF created: {pdf_path}")
        print("📊 Each scene is on its own page for better readability")

        return str(pdf_path)

    except Exception as e:
        print(f"❌ Enhanced PDF generation failed: {e}")
        if "font" in str(e).lower():
            print("💡 Font issue detected. This might be due to missing system fonts.")
        elif "image" in str(e).lower():
            print("💡 Image issue detected. Check if all scene images exist.")
        else:
            print("💡 This might be due to missing system dependencies.")
            print("   On Ubuntu/Debian: sudo apt-get install libpango-1.0-0 libharfbuzz0b libcairo-gobject2")
        return None


def extract_story_data_from_html(html_file_path):
    """
    Extract story data from existing HTML file using BeautifulSoup.

    Args:
        html_file_path (Path): Path to HTML file

    Returns:
        dict: Extracted story data
    """
    try:
        from bs4 import BeautifulSoup
        from bs4.element import Tag
        from datetime import datetime

        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract basic info
        title_elem = soup.find('h2')
        original_prompt = title_elem.get_text().strip('"') if title_elem else "Unknown Story"

        # Extract model info and generation date
        model = 'Unknown'
        generated_at = datetime.now().isoformat()

        generated_info = soup.find(class_='generated-info')
        if generated_info:
            info_text = generated_info.get_text()
            if 'Model:' in info_text:
                model_line = [line for line in info_text.split('\n') if 'Model:' in line]
                if model_line:
                    model = model_line[0].replace('Model:', '').strip()

            if 'Generated on:' in info_text:
                date_line = [line for line in info_text.split('\n') if 'Generated on:' in line]
                if date_line:
                    generated_at = date_line[0].replace('Generated on:', '').strip()

        # Extract story info details
        story_info = soup.find(class_='story-info')
        num_scenes = 0
        if story_info:
            info_text = story_info.get_text()
            if 'Scenes Generated:' in info_text:
                scenes_line = [line for line in info_text.split('\n') if 'Scenes Generated:' in line]
                if scenes_line:
                    try:
                        num_scenes = int(scenes_line[0].split(':')[1].strip())
                    except ValueError:
                        pass

        # Extract scenes
        scenes = []
        scene_divs = soup.find_all(class_='scene')
        print(f"🎬 Processing {len(scene_divs)} scene divs for extraction")

        for i, scene_div in enumerate(scene_divs):
            if not isinstance(scene_div, Tag):
                print(f"❌ Skipping invalid scene div {i}: {type(scene_div)}")
                continue
            # Get scene number
            scene_number_elem = scene_div.find('div', class_='scene-number')
            if scene_number_elem:
                scene_num_text = scene_number_elem.get_text()
                try:
                    scene_number = int(scene_num_text.replace('Scene', '').strip())
                    print(f"📝 Processing scene {scene_number}")
                except ValueError:
                    print(f"❌ Could not parse scene number from: {scene_num_text}")
                    continue
            else:
                print(f"❌ No scene number found in scene div {i}")
                continue

            # Get image
            img_elem = scene_div.find('img', class_='scene-image')
            if img_elem and isinstance(img_elem, Tag) and img_elem.has_attr('src'):
                scenes.append({
                    'type': 'image',
                    'filename': img_elem['src'],
                    'scene_number': scene_number
                })
                print(f"   🖼️  Added image: {img_elem['src']}")

            # Get text
            text_elem = scene_div.find(class_='scene-text')
            text_content = None

            if text_elem:
                text_content = text_elem.get_text().strip()
            else:
                # Alternative: look for div with strong tag containing scene info
                strong_elem = scene_div.find('strong')
                if isinstance(strong_elem, Tag):
                    # Get the parent div of the strong element
                    parent_div = strong_elem.parent
                    if parent_div and parent_div.name == 'div':
                        text_content = parent_div.get_text().strip()
                else:
                    # Last resort: get all text content except scene number
                    all_text = scene_div.get_text().strip()
                    # Remove scene number and image alt text
                    lines = all_text.split('\n')
                    text_lines = [line.strip() for line in lines
                                if line.strip() and
                                not line.strip().startswith(f'Scene {scene_number}') and
                                not line.strip().startswith('alt=')]
                    if text_lines:
                        text_content = ' '.join(text_lines)

            if text_content:
                # Clean up the text
                if text_content.startswith(f'Scene {scene_number}:'):
                    text_content = text_content[len(f'Scene {scene_number}:'):].strip()
                elif text_content.startswith(f'**Scene {scene_number}:**'):
                    text_content = text_content[len(f'**Scene {scene_number}:**'):].strip()

                if text_content:  # Make sure we still have content after cleanup
                    scenes.append({
                        'type': 'text',
                        'content': text_content,
                        'scene_number': scene_number
                    })

        story_data = {
            'scenes': scenes,
            'generated_at': generated_at,
            'model': model,
            'original_prompt': original_prompt,
            'num_scenes': num_scenes or len([s for s in scenes if s['type'] == 'image'])
        }

        return story_data

    except ImportError:
        print("⚠️  BeautifulSoup not available, using basic extraction")
        return None
    except Exception as e:
        print(f"❌ Error extracting from HTML: {e}")
        return None
def regenerate_existing_story_pdf(story_dir_path):
    """
    Regenerate PDF for an existing story with enhanced formatting.

    Args:
        story_dir_path (str): Path to story directory

    Returns:
        str: Path to new PDF or None if failed
    """
    story_dir = Path(story_dir_path)

    if not story_dir.exists():
        print(f"❌ Story directory not found: {story_dir}")
        return None

    # Look for story metadata first
    metadata_file = story_dir / "story_metadata.json"
    if metadata_file.exists():
        import json
        try:
            with open(metadata_file, 'r') as f:
                story_data = json.load(f)
            print(f"📖 Found story metadata: {story_data.get('original_prompt', 'Unknown')}")
        except Exception as e:
            print(f"❌ Error reading metadata: {e}")
            story_data = None
    else:
        story_data = None

    # If no metadata, try to extract from HTML
    if not story_data:
        html_files = list(story_dir.glob("*.html"))
        if not html_files:
            print("❌ No HTML files found in story directory")
            return None

        # Find the original HTML file (not the print version)
        original_html = None
        for html_file in html_files:
            if not html_file.name.endswith('_print.html'):
                original_html = html_file
                break

        if not original_html:
            print("⚠️  No original HTML file found, using first available")
            original_html = html_files[0]

        print(f"📖 Extracting story data from HTML file: {original_html.name}")
        story_data = extract_story_data_from_html(original_html)

        if not story_data:
            print("⚠️  HTML extraction failed, using basic reconstruction...")
            # Fallback to basic reconstruction
            story_data = {
                'scenes': [],
                'generated_at': datetime.now().isoformat(),
                'model': 'imagen-3.0-generate-001',
                'original_prompt': story_dir.name.replace('story_', '').replace('_', ' '),
                'num_scenes': len(list(story_dir.glob("scene_*.png")))
            }

            # Add scenes from images only
            scene_images = sorted(story_dir.glob("scene_*.png"))
            for i, img_path in enumerate(scene_images):
                story_data['scenes'].append({
                    'type': 'image',
                    'filename': img_path.name,
                    'path': str(img_path),
                    'scene_number': i + 1,
                    'part_index': i
                })

    # Generate enhanced PDF
    return create_enhanced_pdf(story_data, story_dir)

if __name__ == "__main__":
    """Test the enhanced PDF generator with an existing story"""
    import sys

    if len(sys.argv) > 1:
        story_path = sys.argv[1]
        print(f"🔄 Regenerating PDF for: {story_path}")
        result = regenerate_existing_story_pdf(story_path)
        if result:
            print(f"✅ Enhanced PDF created: {result}")
        else:
            print("❌ Failed to create enhanced PDF")
    else:
        print("Usage: python enhanced_pdf_generator.py <story_directory_path>")
        print("Example: python enhanced_pdf_generator.py generated_stories/story_20250528_165909")
