import base64
import os
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any

import pytesseract
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from aworld.config.conf import AgentConfig
from aworld.logs.util import Color
from aworld.models.llm import call_llm_model, get_llm_model
from aworld.models.model_response import ModelResponse
from examples.gaia.mcp_collections.base import ActionArguments, ActionCollection, ActionResponse


class ImageMetadata(BaseModel):
    """Metadata extracted from image processing."""

    file_name: str = Field(description="Original image file name")
    file_size: int = Field(description="File size in bytes")
    file_type: str = Field(description="Image file type/extension")
    absolute_path: str = Field(description="Absolute path to the image file")
    width: int | None = Field(default=None, description="Image width in pixels")
    height: int | None = Field(default=None, description="Image height in pixels")
    mode: str | None = Field(default=None, description="Image color mode (RGB, RGBA, L, etc.)")
    format: str | None = Field(default=None, description="Image format (JPEG, PNG, etc.)")
    has_transparency: bool = Field(default=False, description="Whether image has transparency")
    processing_time: float = Field(description="Time taken to process the image in seconds", exclude=True)
    output_files: list[str] = Field(default_factory=list, description="Paths to generated output files")
    extracted_text: str | None = Field(default=None, description="Text extracted via OCR")
    analysis_result: str | None = Field(default=None, description="AI analysis result")
    compression_ratio: float | None = Field(default=None, description="Compression ratio if optimized")
    output_format: str = Field(description="Format of the processed output")


class ImageCollection(ActionCollection):
    """MCP service for comprehensive image processing and analysis.

    Supports various image operations including:
    - Metadata extraction
    - OCR (Optical Character Recognition)
    - AI-powered image analysis and reasoning
    """

    def __init__(self, arguments: ActionArguments) -> None:
        super().__init__(arguments)
        self._image_output_dir = self.workspace / "processed_images"
        self._image_output_dir.mkdir(exist_ok=True)

        # Supported image formats
        self.supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".bmp",
            ".tiff",
            ".tif",
            ".ico",
            ".svg",
        }

        self._llm_config = AgentConfig(
            llm_provider="openai",
            llm_model_name=os.getenv("IMAGE_LLM_MODEL_NAME", "gpt-4o"),
            llm_api_key=os.getenv("IMAGE_LLM_API_KEY"),
            llm_base_url=os.getenv("IMAGE_LLM_BASE_URL"),
        )

        self._color_log("Image Processing Service initialized", Color.green, "debug")
        self._color_log(f"Image output directory: {self._image_output_dir}", Color.blue, "debug")

    def _load_image(self, file_path: Path) -> Image.Image:
        """Load image from file path.

        Args:
            file_path: Path to the image file

        Returns:
            PIL Image object
        """
        try:
            return Image.open(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {file_path}: {str(e)}") from e

    def _get_image_metadata(self, image: Image.Image, file_path: Path) -> dict[str, Any]:
        """Extract metadata from PIL Image object.

        Args:
            image: PIL Image object
            file_path: Path to the original file

        Returns:
            Dictionary containing image metadata
        """
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format or file_path.suffix.upper().lstrip("."),
            "has_transparency": image.mode in ("RGBA", "LA") or "transparency" in image.info,
        }

    def _optimize_image(self, image: Image.Image, max_size: tuple[int, int] | None = None) -> Image.Image:
        """Optimize image for size and quality.

        Args:
            image: PIL Image object
            max_size: Maximum dimensions (width, height)

        Returns:
            Optimized PIL Image object
        """
        optimized = image.copy()

        # Resize if max_size specified
        if max_size:
            optimized.thumbnail(max_size, Image.Resampling.LANCZOS)

        return optimized

    def _perform_ocr(self, image: Image.Image) -> str:
        """Perform OCR on image to extract text.

        Args:
            image: PIL Image object

        Returns:
            Extracted text string
        """
        try:
            return pytesseract.image_to_string(image).strip()
        except ImportError:
            return "OCR not available - pytesseract not installed"
        except Exception as e:
            return f"OCR failed: {str(e)}"

    def _analyze_with_ai(self, image_base64: str, task: str) -> str:
        """Analyze image using AI model.

        Args:
            image_base64: Base64 encoded image
            task: Analysis task description

        Returns:
            AI analysis result
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"role": "text", "content": task},
                        {"type": "image_url", "image_url": {"url": image_base64}},
                    ],
                },
            ]
            response: ModelResponse = call_llm_model(
                llm_model=get_llm_model(conf=self._llm_config),
                messages=messages,
                temperature=float(os.getenv("LLM_TEMPERATURE", "1.0")),
            )
            self._color_log(f"{response.content=}", Color.green)
            return response.content
        except Exception as e:
            return f"AI analysis failed: {str(e)}"

    def _image_to_base64(self, image: Image.Image, output_format: str = "JPEG") -> str:
        """Convert PIL Image to base64 string.

        Args:
            image: PIL Image object
            output_format: Output format (JPEG, PNG, etc.)

        Returns:
            Base64 encoded image string
        """
        buffer = BytesIO()

        # Convert RGBA or P to RGB for JPEG
        if output_format.upper() == "JPEG":
            if image.mode in ("RGBA", "LA"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
                image = background
            elif image.mode == "P":
                image = image.convert("RGB")

        image.save(buffer, format=output_format, quality=85 if output_format.upper() == "JPEG" else None)

        mime_type = f"image/{output_format.lower()}"
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return f"data:{mime_type};base64,{img_base64}"

    def mcp_extract_text_ocr(
        self,
        file_path: str = Field(description="Path to the image file for OCR"),
        language: str = Field(default="eng", description="OCR language code (e.g., 'eng', 'spa', 'fra')"),
        preprocess: bool = Field(default=True, description="Whether to preprocess image for better OCR"),
    ) -> ActionResponse:
        """Extract text from images using Optical Character Recognition (OCR).

        This tool uses Tesseract OCR to extract text content from images,
        with optional preprocessing to improve recognition accuracy.

        Args:
            file_path: Path to the image file
            language: OCR language for better recognition
            preprocess: Whether to enhance image for OCR

        Returns:
            ActionResponse with extracted text and metadata
        """
        try:
            # Handle FieldInfo objects
            if isinstance(file_path, FieldInfo):
                file_path = file_path.default
            if isinstance(language, FieldInfo):
                language = language.default
            if isinstance(preprocess, FieldInfo):
                preprocess = preprocess.default

            start_time = time.time()

            # Validate input file
            file_path: Path = self._validate_file_path(file_path)
            self._color_log(f"Performing OCR on: {file_path.name}", Color.cyan)

            # Load image
            image = self._load_image(file_path)
            original_metadata = self._get_image_metadata(image, file_path)

            # Preprocess image for better OCR if requested
            if preprocess:
                # Convert to grayscale
                if image.mode != "L":
                    image = image.convert("L")

                # Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.0)

                # Apply slight sharpening
                image = image.filter(ImageFilter.SHARPEN)

            # Perform OCR
            extracted_text = self._perform_ocr(image)
            processing_time = time.time() - start_time

            # Count words and characters
            word_count = len(extracted_text.split()) if extracted_text else 0
            char_count = len(extracted_text) if extracted_text else 0

            # Create metadata object
            image_metadata = ImageMetadata(
                file_name=file_path.name,
                file_size=file_path.stat().st_size,
                file_type=file_path.suffix.lower(),
                absolute_path=str(file_path.absolute()),
                width=original_metadata["width"],
                height=original_metadata["height"],
                mode=original_metadata["mode"],
                format=original_metadata["format"],
                has_transparency=original_metadata["has_transparency"],
                processing_time=processing_time,
                output_files=[],
                extracted_text=extracted_text,
                output_format="ocr_text",
            )

            if extracted_text:
                result_message = (
                    f"OCR Results for {file_path.name}:\n\n"
                    f"**Extracted Text:**\n{extracted_text}\n\n"
                    f"**Statistics:**\n"
                    f"- Words: {word_count}\n"
                    f"- Characters: {char_count}\n"
                    f"- Language: {language}\n"
                    f"- Processing time: {processing_time:.2f}s"
                )
            else:
                result_message = (
                    f"No text detected in {file_path.name}."
                    " The image may not contain readable text or OCR preprocessing may be needed."
                )

            self._color_log(f"OCR completed: {word_count} words extracted in {processing_time:.2f}s", Color.green)

            return ActionResponse(success=True, message=result_message, metadata=image_metadata.model_dump())

        except Exception as e:
            self.logger.error(f"OCR failed: {str(e)}: {traceback.format_exc()}")
            return ActionResponse(
                success=False,
                message=f"OCR failed: {str(e)}",
                metadata={"error_type": "ocr_error"},
            )

    def mcp_analyze_image_ai(
        self,
        file_path: str = Field(description="Path to the image file for AI analysis"),
        task: str = Field(
            default="Describe what you see in this image",
            description="Specific analysis task or question about the image",
        ),
    ) -> ActionResponse:
        """Analyze image content using AI vision models.

        This tool uses advanced AI models to analyze and describe image content,
        answer questions about images, or perform specific visual reasoning tasks.

        Args:
            file_path: Path to the image file
            task: Specific analysis task or question

        Returns:
            ActionResponse with AI analysis results and metadata
        """
        try:
            # Handle FieldInfo objects
            if isinstance(file_path, FieldInfo):
                file_path = file_path.default
            if isinstance(task, FieldInfo):
                task = task.default

            start_time = time.time()

            # Validate input file
            file_path: Path = self._validate_file_path(file_path)
            self._color_log(f"Analyzing image with AI: {file_path.name}", Color.cyan)

            # Load image
            image = self._load_image(file_path)
            original_metadata = self._get_image_metadata(image, file_path)

            # Convert to base64 for AI analysis
            image_base64 = self._image_to_base64(image, "JPEG")

            # Perform AI analysis
            analysis_result = self._analyze_with_ai(image_base64, task)
            processing_time = time.time() - start_time

            # Create metadata object
            metadata_dict = {
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix.lower(),
                "absolute_path": str(file_path.absolute()),
                "width": original_metadata["width"],
                "height": original_metadata["height"],
                "mode": original_metadata["mode"],
                "format": original_metadata["format"],
                "has_transparency": original_metadata["has_transparency"],
                "processing_time": processing_time,
                "output_files": [],
                "analysis_result": analysis_result,
                "output_format": "ai_analysis",
            }

            image_metadata = ImageMetadata(**metadata_dict)

            result_message = (
                f"AI Analysis Results for {file_path.name}:\n\n"
                f"**Task:** {task}\n\n"
                f"**Analysis:**\n{analysis_result}\n\n"
                f"**Image Info:**\n"
                f"- Dimensions: {original_metadata['width']}x{original_metadata['height']}\n"
                f"- Format: {original_metadata['format']}\n"
                f"- Processing time: {processing_time:.2f}s"
            )

            self._color_log(f"AI analysis completed in {processing_time:.2f}s", Color.green)

            return ActionResponse(success=True, message=result_message, metadata=image_metadata.model_dump())

        except Exception as e:
            self.logger.error(f"AI image analysis failed: {str(e)}: {traceback.format_exc()}")
            return ActionResponse(
                success=False,
                message=f"AI image analysis failed: {str(e)}",
                metadata={"error_type": "ai_analysis_error"},
            )

    def mcp_get_image_metadata(
        self,
        file_path: str = Field(description="Path to the image file to analyze"),
    ) -> ActionResponse:
        """Extract comprehensive metadata from image files.

        This tool analyzes image files and extracts detailed metadata including
        dimensions, format, color mode, file size, and other technical information.

        Args:
            file_path: Path to the image file to analyze

        Returns:
            ActionResponse with detailed image metadata
        """
        try:
            if isinstance(file_path, FieldInfo):
                file_path = file_path.default

            start_time = time.time()

            # Validate input file
            file_path: Path = self._validate_file_path(file_path)
            self._color_log(f"Extracting metadata from: {file_path.name}", Color.cyan)

            # Load image
            image = self._load_image(file_path)
            metadata = self._get_image_metadata(image, file_path)
            processing_time = time.time() - start_time

            # Get file statistics
            file_stats = file_path.stat()

            # Create metadata object
            image_metadata = ImageMetadata(
                file_name=file_path.name,
                file_size=file_stats.st_size,
                file_type=file_path.suffix.lower(),
                absolute_path=str(file_path.absolute()),
                width=metadata["width"],
                height=metadata["height"],
                mode=metadata["mode"],
                format=metadata["format"],
                has_transparency=metadata["has_transparency"],
                processing_time=processing_time,
                output_files=[],
                output_format="metadata",
            )

            # Calculate additional info
            aspect_ratio = metadata["width"] / metadata["height"] if metadata["height"] > 0 else 0
            megapixels = (metadata["width"] * metadata["height"]) / 1_000_000

            result_message = (
                f"Image Metadata for {file_path.name}:\n"
                f"Dimensions: {metadata['width']}x{metadata['height']} pixels\n"
                f"Aspect Ratio: {aspect_ratio:.2f}:1\n"
                f"Megapixels: {megapixels:.2f} MP\n"
                f"Color Mode: {metadata['mode']}\n"
                f"Format: {metadata['format']}\n"
                f"Has Transparency: {metadata['has_transparency']}\n"
                f"File Size: {file_stats.st_size / 1024 / 1024:.2f} MB\n"
                f"File Type: {file_path.suffix.upper()}"
            )

            self._color_log(f"Metadata extraction completed in {processing_time:.2f}s", Color.green)

            return ActionResponse(success=True, message=result_message, metadata=image_metadata.model_dump())

        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {str(e)}: {traceback.format_exc()}")
            return ActionResponse(
                success=False,
                message=f"Metadata extraction failed: {str(e)}",
                metadata={"error_type": "metadata_error"},
            )


if __name__ == "__main__":
    load_dotenv()
    # Default arguments for testing
    args = ActionArguments(
        name="image_analysis_service",
        transport="stdio",
        workspace=os.getenv("AWORLD_WORKSPACE", "~"),
    )
    # Initialize and run the image analysis service
    try:
        service = ImageCollection(args)
        service.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
