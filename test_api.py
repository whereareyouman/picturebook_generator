#!/usr/bin/env python3
"""
Simple test script to validate the Gemini API connection and basic functionality.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

def test_api():
    """Test basic API functionality."""
    print("🔍 Testing Google Gemini API connection...")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("❌ No API key found in .env file")
        return False
    
    try:
        # Initialize client
        client = genai.Client(api_key=api_key)
        print("✅ Client initialized successfully")
        
        # Test simple text generation
        print("🧪 Testing text generation...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello! Please respond with 'API test successful' if you can hear me."
        )
        
        if response and response.candidates:
            text_content = response.candidates[0].content.parts[0].text
            print(f"✅ Text response: {text_content}")
            
            # Test image generation (Gemini image model)
            print("🖼️  Testing image generation...")
            try:
                image_response = client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents="Generate a simple cartoon cat sitting on a sunny windowsill",
                    config=types.GenerateContentConfig(
                        response_modalities=["Text", "Image"]
                    )
                )

                if image_response and image_response.candidates:
                    parts = image_response.candidates[0].content.parts
                    print(f"✅ Image generation response has {len(parts)} parts")

                    found_image = False
                    for i, part in enumerate(parts):
                        if hasattr(part, "text") and part.text:
                            print(f"   Part {i}: Text content found")
                        elif hasattr(part, "inline_data") and part.inline_data:
                            print(f"   Part {i}: Image data found ✅")
                            found_image = True

                            image_bytes = part.inline_data.data
                            out_name = "test_generated_image"
                            if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                                out_path = f"{out_name}.png"
                            elif image_bytes.startswith(b"\xff\xd8\xff"):
                                out_path = f"{out_name}.jpg"
                            else:
                                out_path = f"{out_name}.bin"

                            with open(out_path, "wb") as f:
                                f.write(image_bytes)

                            print(f"✅ 图片已保存: {out_path}")
                            break
                        else:
                            print(f"   Part {i}: Unknown part type")

                    if found_image:
                        return True

                    print("❌ 图像生成失败：响应中没有图片数据 (inline_data)")
                    return False

                print("❌ Image generation failed - no response")
                return False
            except Exception as e:
                print(f"❌ 图像生成出错: {e}")
                return False
        else:
            print("❌ Text generation failed - no response")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    if success:
        print("\n🎉 All tests passed! The picture book generator should work now.")
    else:
        print("\n😞 Tests failed. Please check your API key and internet connection.")
