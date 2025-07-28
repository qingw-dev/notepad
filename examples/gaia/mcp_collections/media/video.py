"""
Video MCP Service

This module provides MCP service functionality for video operations including:
- Video content analysis with AI-powered insights
- Video summarization and key point extraction
- Keyframe extraction with scene detection
- Subtitle extraction from video content

It handles various video formats with proper validation, error handling,
and progress tracking while providing LLM-friendly formatted results.
"""

import base64
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from aworld.config.conf import AgentConfig
from aworld.logs.util import Color
from aworld.models.llm import call_llm_model, get_llm_model
from aworld.models.model_response import ModelResponse
from examples.gaia.mcp_collections.base import (
    ActionArguments,
    ActionCollection,
    ActionResponse,
)

from ..utils import get_file_from_source


class VideoAnalysisResult(BaseModel):
    """Video analysis result model with structured data"""

    video_source: str
    analysis_result: str
    frame_count: int
    duration_analyzed: float
    success: bool
    error: str | None = None


class VideoSummaryResult(BaseModel):
    """Video summary result model with structured data"""

    video_source: str
    summary: str
    frame_count: int
    duration_analyzed: float
    success: bool
    error: str | None = None


class KeyframeResult(BaseModel):
    """Keyframe extraction result model with file information"""

    frame_paths: list[str]
    frame_timestamps: list[float]
    output_directory: str
    frame_count: int
    target_time: float
    window_size: float
    success: bool
    error: str | None = None


class VideoMetadata(BaseModel):
    """Metadata for video operation results"""

    operation: str
    video_source: str | None = None
    sample_rate: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    target_time: float | None = None
    window_size: float | None = None
    output_directory: str | None = None
    frame_count: int | None = None
    execution_time: float | None = None
    error_type: str | None = None


class VideoCollection(ActionCollection):
    """MCP service for video operations with AI-powered analysis.

    Provides video processing capabilities including:
    - AI-powered video content analysis
    - Video summarization with key insights
    - Keyframe extraction with scene detection
    - Subtitle extraction from video content
    - LLM-friendly result formatting
    - Error handling and logging
    """

    def __init__(self, arguments: ActionArguments) -> None:
        super().__init__(arguments)

        # Initialize supported video extensions
        self.supported_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

        # Video analysis prompts
        self.video_analyze_prompt = (
            "Input is a sequence of video frames. Given user's task: {task}. "
            "analyze the video content following these steps:\n"
            "1. Temporal sequence understanding\n"
            "2. Motion and action analysis\n"
            "3. Scene context interpretation\n"
            "4. Object and person tracking\n"
        )

        self.video_summarize_prompt = (
            "Input is a sequence of video frames. "
            "Summarize the main content of the video. "
            "Include key points, main topics, and important visual elements. "
        )

        self._color_log("Video service initialized", Color.green, "debug")

    def _get_video_frames(
        self,
        video_source: str,
        sample_rate: int = 2,
        start_time: float = 0,
        end_time: float | None = None,
    ) -> list[dict[str, any]]:
        """Extract frames from video with given sample rate.

        Args:
            video_source: Path or URL to the video file
            sample_rate: Number of frames to sample per second
            start_time: Start time of the video segment in seconds
            end_time: End time of the video segment in seconds

        Returns:
            List of dictionaries containing frame data and timestamp

        Raises:
            ValueError: When video file cannot be opened or is not valid
        """
        try:
            # Get file with validation (only video files allowed)
            file_path, _, _ = get_file_from_source(
                video_source,
                max_size_mb=2500.0,  # 2500MB limit for videos
            )

            # Open video file
            video = cv2.VideoCapture(file_path)  # pylint: disable=E1101
            if not video.isOpened():
                raise ValueError(f"Could not open video file: {file_path}")

            fps = video.get(cv2.CAP_PROP_FPS)  # pylint: disable=E1101
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # pylint: disable=E1101
            video_duration = frame_count / fps

            if end_time is None:
                end_time = video_duration

            if start_time > end_time:
                raise ValueError("Start time cannot be greater than end time.")

            if start_time < 0:
                start_time = 0

            if end_time > video_duration:
                end_time = video_duration

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            all_frames = []
            frames = []

            # Calculate frame interval based on sample rate
            frame_interval = max(1, int(fps / sample_rate))

            # Set the video capture to the start frame
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # pylint: disable=E1101

            for i in range(start_frame, end_frame):
                ret, frame = video.read()
                if not ret:
                    break

                # Convert frame to JPEG format
                _, buffer = cv2.imencode(".jpg", frame)  # pylint: disable=E1101
                frame_data = base64.b64encode(buffer).decode("utf-8")

                # Add data URL prefix for JPEG image
                frame_data = f"data:image/jpeg;base64,{frame_data}"

                all_frames.append({"data": frame_data, "time": i / fps})

            for i in range(0, len(all_frames), frame_interval):
                frames.append(all_frames[i])

            video.release()

            # Clean up temporary file if it was created for a URL
            if (
                file_path != str(Path(video_source).resolve())
                and Path(file_path).exists()
            ):
                Path(file_path).unlink()

            if not frames:
                raise ValueError(
                    f"Could not extract any frames from video: {video_source}"
                )

            return frames

        except Exception as e:
            self._color_log(
                f"Error extracting frames from {video_source}: {str(e)}", Color.red
            )
            raise

    def _create_video_content(
        self, prompt: str, video_frames: list[dict[str, any]]
    ) -> list[dict[str, any]]:
        """Create uniform video format for querying LLM."""
        content = [{"type": "text", "text": prompt}]
        content.extend(
            [
                {"type": "image_url", "image_url": {"url": frame["data"]}}
                for frame in video_frames
            ]
        )
        return content

    def _format_analysis_output(
        self, result: VideoAnalysisResult, format_type: str = "markdown"
    ) -> str:
        """Format video analysis results for LLM consumption.

        Args:
            result: Video analysis result
            format_type: Output format ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not result.success:
            return f"Failed to analyze video: {result.error}"

        if format_type == "json":
            return result.model_dump_json(indent=2)

        elif format_type == "text":
            output_parts = [
                "Video Analysis Results",
                f"Source: {result.video_source}",
                f"Frames Analyzed: {result.frame_count}",
                f"Duration: {result.duration_analyzed:.2f} seconds",
                "",
                "Analysis:",
                result.analysis_result,
            ]
            return "\n".join(output_parts)

        else:  # markdown (default)
            output_parts = [
                "# Video Analysis Results ✅",
                "",
                "## Video Information",
                f"**Source:** `{result.video_source}`",
                f"**Frames Analyzed:** {result.frame_count}",
                f"**Duration:** {result.duration_analyzed:.2f} seconds",
                "",
                "## Analysis Results",
                result.analysis_result,
            ]
            return "\n".join(output_parts)

    def _format_summary_output(
        self, result: VideoSummaryResult, format_type: str = "markdown"
    ) -> str:
        """Format video summary results for LLM consumption.

        Args:
            result: Video summary result
            format_type: Output format ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not result.success:
            return f"Failed to summarize video: {result.error}"

        if format_type == "json":
            return result.model_dump_json(indent=2)

        elif format_type == "text":
            output_parts = [
                "Video Summary",
                f"Source: {result.video_source}",
                f"Frames Analyzed: {result.frame_count}",
                f"Duration: {result.duration_analyzed:.2f} seconds",
                "",
                "Summary:",
                result.summary,
            ]
            return "\n".join(output_parts)

        else:  # markdown (default)
            output_parts = [
                "# Video Summary ✅",
                "",
                "## Video Information",
                f"**Source:** `{result.video_source}`",
                f"**Frames Analyzed:** {result.frame_count}",
                f"**Duration:** {result.duration_analyzed:.2f} seconds",
                "",
                "## Summary",
                result.summary,
            ]
            return "\n".join(output_parts)

    def _format_keyframe_output(
        self, result: KeyframeResult, format_type: str = "markdown"
    ) -> str:
        """Format keyframe extraction results for LLM consumption.

        Args:
            result: Keyframe extraction result
            format_type: Output format ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not result.success:
            return f"Failed to extract keyframes: {result.error}"

        if format_type == "json":
            return result.model_dump_json(indent=2)

        elif format_type == "text":
            output_parts = [
                "Keyframe Extraction Results",
                f"Target Time: {result.target_time}s",
                f"Window Size: {result.window_size}s",
                f"Frames Extracted: {result.frame_count}",
                f"Output Directory: {result.output_directory}",
                "",
                "Frame Files:",
            ]
            for i, (path, timestamp) in enumerate(
                zip(result.frame_paths, result.frame_timestamps), 1
            ):
                output_parts.append(f"{i}. {path} (at {timestamp:.2f}s)")

            return "\n".join(output_parts)

        else:  # markdown (default)
            output_parts = [
                "# Keyframe Extraction Results ✅",
                "",
                "## Extraction Parameters",
                f"**Target Time:** {result.target_time}s",
                f"**Window Size:** {result.window_size}s",
                f"**Frames Extracted:** {result.frame_count}",
                f"**Output Directory:** `{result.output_directory}`",
                "",
                "## Extracted Frames",
            ]

            for i, (path, timestamp) in enumerate(
                zip(result.frame_paths, result.frame_timestamps), 1
            ):
                output_parts.append(f"{i}. `{path}` (at {timestamp:.2f}s)")

            return "\n".join(output_parts)

    def _analyze_frame_chunk(
        self, chunk_data: tuple[int, list, str]
    ) -> tuple[int, str]:
        """Analyze a chunk of video frames using LLM.

        Args:
            chunk_data: Tuple containing (chunk_index, frames, question)

        Returns:
            Tuple of (chunk_index, analysis_result)
        """
        chunk_index, frames, question = chunk_data

        try:
            content = self._create_video_content(
                self.video_analyze_prompt.format(task=question), frames
            )
            inputs = [{"role": "user", "content": content}]

            response: ModelResponse = call_llm_model(
                get_llm_model(
                    conf=AgentConfig(
                        llm_provider="openai",
                        llm_model_name=os.getenv("VIDEO_LLM_MODEL_NAME"),
                        llm_api_key=os.getenv("VIDEO_LLM_API_KEY"),
                        llm_base_url=os.getenv("VIDEO_LLM_BASE_URL"),
                    )
                ),
                inputs,
                temperature=float(os.getenv("VIDEO_LLM_TEMPERATURE", "1.0")),
            )
            analysis_result = response.content
            self._color_log(
                f"✅ Completed analysis for chunk {chunk_index + 1}", Color.green
            )

        except Exception as e:
            self._color_log(
                f"❌ LLM analysis error for chunk {chunk_index + 1}: {str(e)}",
                Color.yellow,
            )
            analysis_result = (
                f"Analysis failed for video segment {chunk_index + 1}: {str(e)}"
            )

        return chunk_index, analysis_result

    async def mcp_analyze_video(
        self,
        video_url: str = Field(description="Path or URL to the video file to analyze"),
        question: str = Field(description="Question or task for video analysis"),
        sample_rate: float = Field(
            default=1.0, description="Frame sampling rate (frames per second)"
        ),
        start_time: float = Field(default=0.0, description="Start time in seconds"),
        end_time: float | None = Field(
            default=None, description="End time in seconds (None for full video)"
        ),
        output_format: str = Field(
            default="markdown",
            description="Output format: 'markdown', 'json', or 'text'",
        ),
        max_workers: int = Field(
            default=4, description="Maximum number of parallel workers for analysis"
        ),
    ) -> ActionResponse:
        """Analyze video content using AI with parallel processing.

        This tool provides comprehensive video analysis capabilities including:
        - Content understanding and description
        - Object and scene detection
        - Action and movement analysis
        - Temporal event tracking
        - Question-answering about video content
        - Parallel processing for faster analysis

        Args:
            video_url: Path or URL to the video file
            question: Specific question or analysis task
            sample_rate: Frame sampling rate for analysis
            start_time: Start time of the video segment in seconds
            end_time: End time of the video segment in seconds
            output_format: Format for the response output
            max_workers: Maximum number of parallel workers

        Returns:
            ActionResponse with video analysis results and metadata
        """
        start_exec_time = time.time()

        try:
            # Validate video file
            video_path = self._validate_file_path(video_url)

            self._color_log(f"🎬 Analyzing video: {video_url}", Color.cyan)
            self._color_log(f"📋 Question: {question}", Color.blue)

            # Extract video frames
            video_frames = self._get_video_frames(
                str(video_path), sample_rate, start_time, end_time
            )
            self._color_log(f"📸 Extracted {len(video_frames)} frames", Color.blue)

            # Process frames in chunks of 64 frames for parallel analysis
            chunk_size = 128
            chunks = []

            # Create chunks of `chunk_size` continuous frames
            for i in range(0, len(video_frames), chunk_size):
                chunk_frames = video_frames[i : i + chunk_size]
                chunks.append((i // chunk_size, chunk_frames, question))

            self._color_log(
                f"🔄 Processing {len(chunks)} chunks with {max_workers} parallel workers",
                Color.blue,
            )

            # Process chunks in parallel
            all_results = [None] * len(chunks)  # Pre-allocate to maintain order

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunk analysis tasks
                future_to_chunk = {
                    executor.submit(self._analyze_frame_chunk, chunk_data): chunk_data[
                        0
                    ]
                    for chunk_data in chunks
                }

                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_index, result = future.result()
                        all_results[chunk_index] = (
                            f"Result of video part {chunk_index + 1}: {result}"
                        )
                    except Exception as e:
                        chunk_index = future_to_chunk[future]
                        self._color_log(
                            f"❌ Error processing chunk {chunk_index + 1}: {str(e)}",
                            Color.red,
                        )
                        all_results[chunk_index] = (
                            f"Result of video part {chunk_index + 1}: Analysis failed - {str(e)}"
                        )

            # Filter out None results and join
            analysis_result = "\n".join(
                [result for result in all_results if result is not None]
            )
            duration_analyzed = (
                end_time - start_time if end_time else len(video_frames) / sample_rate
            )

            # Create result
            result = VideoAnalysisResult(
                video_source=video_url,
                analysis_result=analysis_result,
                frame_count=len(video_frames),
                duration_analyzed=duration_analyzed,
                success=True,
                error=None,
            )

            # Format output for LLM
            message = self._format_analysis_output(result, output_format)
            execution_time = time.time() - start_exec_time

            # Create metadata
            metadata = {
                "video_source": video_url,
                "frame_count": len(video_frames),
                "chunks_processed": len(chunks),
                "chunk_size": chunk_size,
                "parallel_workers": max_workers,
                "duration_analyzed": duration_analyzed,
                "sample_rate": sample_rate,
                "start_time": start_time,
                "end_time": end_time,
                "execution_time": execution_time,
                "output_format": output_format,
                "success": True,
            }

            self._color_log(
                f"✅ Video analysis completed in {execution_time:.2f}s "
                f"({len(chunks)} chunks, {len(video_frames)} frames)",
                Color.green,
            )

            return ActionResponse(success=True, message=message, metadata=metadata)

        except Exception as e:
            execution_time = time.time() - start_exec_time
            error_msg = f"Video analysis failed: {str(e)}"
            self._color_log(f"❌ {error_msg}", Color.red)
            self.logger.error(f"{error_msg}: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata={
                    "video_source": video_url,
                    "execution_time": execution_time,
                    "error": str(e),
                    "success": False,
                },
            )

    async def mcp_summarize_video(
        self,
        video_url: str = Field(
            description="The input video filepath or URL to summarize."
        ),
        sample_rate: int = Field(
            default=1, description="Sample n frames per second (default: 1)."
        ),
        start_time: float = Field(
            default=0,
            description="Start time of the video segment in seconds (default: 0).",
        ),
        end_time: float | None = Field(
            default=None,
            description="End time of the video segment in seconds (default: None).",
        ),
        output_format: str = Field(
            default="markdown",
            description="Output format: 'markdown', 'json', or 'text' (default: markdown).",
        ),
    ) -> ActionResponse:
        """Summarize the main content of a video using AI analysis.

        This tool provides AI-powered video summarization with:
        - Key point extraction
        - Main topic identification
        - Important visual element recognition
        - LLM-optimized result formatting

        Args:
            video_url: The input video filepath or URL to summarize
            sample_rate: Sample n frames per second
            start_time: Start time of the video segment in seconds
            end_time: End time of the video segment in seconds
            output_format: Format for the response output

        Returns:
            ActionResponse with video summary results and metadata
        """
        start_exec_time = time.time()

        try:
            # Validate video file
            video_path = self._validate_file_path(video_url)

            self._color_log(f"🎬 Summarizing video: {video_url}", Color.cyan)

            # Extract video frames
            video_frames = self._get_video_frames(
                str(video_path), sample_rate, start_time, end_time
            )
            self._color_log(f"📸 Extracted {len(video_frames)} frames", Color.blue)

            # Process frames in larger chunks for summarization
            interval = 490
            frame_nums = 500
            all_results = []

            for i in range(0, len(video_frames), interval):
                cur_frames = video_frames[i : i + frame_nums]
                content = self._create_video_content(
                    self.video_summarize_prompt, cur_frames
                )
                inputs = [{"role": "user", "content": content}]

                try:
                    response: ModelResponse = call_llm_model(
                        get_llm_model(
                            conf=AgentConfig(
                                llm_provider="openai",
                                llm_model_name=os.getenv(
                                    "VIDEO_LLM_MODEL_NAME", "gpt-4o"
                                ),
                                llm_api_key=os.getenv("VIDEO_LLM_API_KEY"),
                                llm_base_url=os.getenv("VIDEO_LLM_BASE_URL"),
                            )
                        ),
                        inputs,
                        temperature=float(os.getenv("VIDEO_LLM_TEMPERATURE", "1.0")),
                    )
                    cur_summary = response.content
                except Exception as e:
                    self._color_log(
                        f"LLM summary error for chunk {i // interval + 1}: {str(e)}",
                        Color.yellow,
                    )
                    cur_summary = (
                        f"Summary failed for video segment {i // interval + 1}"
                    )

                all_results.append(
                    f"Summary of video part {i // interval + 1}: {cur_summary}"
                )

                if i + frame_nums >= len(video_frames):
                    break

            summary_result = "\n".join(all_results)
            duration_analyzed = (
                end_time - start_time if end_time else len(video_frames) / sample_rate
            )

            # Create result
            result = VideoSummaryResult(
                video_source=video_url,
                summary=summary_result,
                frame_count=len(video_frames),
                duration_analyzed=duration_analyzed,
                success=True,
                error=None,
            )

            # Format output for LLM
            message = self._format_summary_output(result, output_format)
            execution_time = time.time() - start_exec_time

            # Create metadata
            metadata = VideoMetadata(
                operation="summarize",
                video_source=video_url,
                sample_rate=sample_rate,
                start_time=start_time,
                end_time=end_time,
                frame_count=len(video_frames),
                execution_time=execution_time,
            ).model_dump()

            self._color_log(
                "✅ Video summarization completed successfully", Color.green
            )
            return ActionResponse(success=True, message=message, metadata=metadata)

        except Exception as e:
            error_msg = str(e)
            self._color_log(
                f"❌ Video summarization error: {traceback.format_exc()}", Color.red
            )

            # Format error for LLM
            message = f"Failed to summarize video: {error_msg}"
            execution_time = time.time() - start_exec_time

            # Create metadata
            metadata = VideoMetadata(
                operation="summarize",
                video_source=video_url,
                error_type="summarization_failure",
                execution_time=execution_time,
            ).model_dump()

            return ActionResponse(success=False, message=message, metadata=metadata)

    async def mcp_extract_keyframes(
        self,
        video_path: str = Field(description="The input video filepath or URL."),
        target_time: int = Field(
            description="The specific time point for extraction (in seconds), centered within the window_size."
        ),
        window_size: int = Field(
            default=5,
            description="The window size for extraction (in seconds, default: 5).",
        ),
        output_dir: str = Field(
            default=None,
            description="Directory where extracted frames will be saved (default: workspace/keyframes).",
        ),
        output_format: str = Field(
            default="markdown",
            description="Output format: 'markdown', 'json', or 'text' (default: markdown).",
        ),
    ) -> ActionResponse:
        """Extract key frames around a target time with scene detection.

        This tool provides keyframe extraction with:
        - Scene detection for significant frame changes
        - Configurable time windows
        - Automatic output directory management
        - LLM-optimized result formatting

        Args:
            video_path: The input video filepath or URL
            target_time: Specific time point (in seconds) to extract frames around
            window_size: Time window (in seconds) centered on target_time
            cleanup: Whether to delete the original video file after processing
            output_dir: Directory where extracted frames will be saved
            output_format: Format for the response output

        Returns:
            ActionResponse with keyframe extraction results and metadata
        """
        start_exec_time = time.time()

        try:
            # Validate video file
            validated_path = self._validate_file_path(video_path)

            # Set default output directory
            output_dir = (
                str(self.workspace / "keyframes") if output_dir is None else output_dir
            )

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            self._color_log(f"🎬 Extracting keyframes from: {video_path}", Color.cyan)
            self._color_log(
                f"🎯 Target time: {target_time}s, Window: {window_size}s", Color.blue
            )

            # Extract keyframes with scene detection
            frames, frame_times = self._extract_keyframes_with_scene_detection(
                str(validated_path), target_time, window_size
            )

            # Save frames to disk
            frame_paths, frame_timestamps = self._save_keyframes(
                frames, frame_times, str(output_path)
            )

            # Cleanup if requested
            # if cleanup and validated_path.exists():
            #     validated_path.unlink()
            #     self._color_log("🗑️ Cleaned up original video file", Color.yellow)

            # Create result
            result = KeyframeResult(
                frame_paths=frame_paths,
                frame_timestamps=frame_timestamps,
                output_directory=str(output_path),
                frame_count=len(frame_paths),
                target_time=float(target_time),
                window_size=float(window_size),
                success=True,
                error=None,
            )

            # Format output for LLM
            message = self._format_keyframe_output(result, output_format)
            execution_time = time.time() - start_exec_time

            # Create metadata
            metadata = VideoMetadata(
                operation="extract_keyframes",
                video_source=video_path,
                target_time=float(target_time),
                window_size=float(window_size),
                output_directory=str(output_path),
                frame_count=len(frame_paths),
                execution_time=execution_time,
            ).model_dump()

            self._color_log(
                f"✅ Extracted {len(frame_paths)} keyframes successfully", Color.green
            )
            return ActionResponse(success=True, message=message, metadata=metadata)

        except Exception as e:
            error_msg = str(e)
            self._color_log(
                f"❌ Keyframe extraction error: {traceback.format_exc()}", Color.red
            )

            # Format error for LLM
            message = f"Failed to extract keyframes: {error_msg}"
            execution_time = time.time() - start_exec_time

            # Create metadata
            metadata = VideoMetadata(
                operation="extract_keyframes",
                video_source=video_path,
                target_time=float(target_time) if target_time else None,
                window_size=float(window_size) if window_size else None,
                error_type="keyframe_extraction_failure",
                execution_time=execution_time,
            ).model_dump()

            return ActionResponse(success=False, message=message, metadata=metadata)

    def _extract_keyframes_with_scene_detection(
        self, video_path: str, target_time: int, window_size: int
    ) -> tuple[list[any], list[float]]:
        """Extract key frames around the target time with scene detection.

        Args:
            video_path: Path to the video file
            target_time: Target time in seconds
            window_size: Window size in seconds

        Returns:
            Tuple of (frames, frame_times)
        """
        cap = cv2.VideoCapture(video_path)  # pylint: disable=E1101
        fps = cap.get(cv2.CAP_PROP_FPS)  # pylint: disable=E1101

        # Calculate frame numbers for the time window
        start_frame = int((target_time - window_size / 2) * fps)
        end_frame = int((target_time + window_size / 2) * fps)
        total_frames_in_window = end_frame - start_frame

        max_frames = 384  # Maximum allowed frames to prevent memory issues

        # Calculate sampling interval for even distribution
        if total_frames_in_window <= max_frames:
            # If total frames is within limit, use scene detection normally
            frame_interval = 1
            use_scene_detection = True
        else:
            # If exceeds limit, sample evenly across the window
            frame_interval = total_frames_in_window // max_frames
            use_scene_detection = False  # Skip scene detection for even sampling

        frames = []
        frame_times = []

        # Set video position to start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))  # pylint: disable=E1101

        prev_frame = None
        frame_count = 0

        while cap.isOpened() and len(frames) < max_frames:
            frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)  # pylint: disable=E1101
            if frame_pos >= end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if use_scene_detection:
                # Use original scene detection logic
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # pylint: disable=E1101

                # If this is the first frame, save it
                if prev_frame is None:
                    frames.append(frame)
                    frame_times.append(frame_pos / fps)
                else:
                    # Calculate difference between current and previous frame
                    diff = cv2.absdiff(gray, prev_frame)  # pylint: disable=E1101
                    mean_diff = np.mean(diff)

                    # If significant change detected, save frame
                    if mean_diff > 20:  # Threshold for scene change
                        frames.append(frame)
                        frame_times.append(frame_pos / fps)

                prev_frame = gray
            else:
                # Use even sampling for large windows
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                    frame_times.append(frame_pos / fps)

                frame_count += 1

        cap.release()
        return frames, frame_times

    def _save_keyframes(
        self, frames: list[any], frame_times: list[float], output_dir: str
    ) -> tuple[list[str], list[float]]:
        """Save extracted frames to disk.

        Args:
            frames: List of frame objects
            frame_times: List of frame timestamps
            output_dir: Output directory path

        Returns:
            Tuple of (saved_paths, saved_timestamps)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        saved_timestamps = []

        for _, (frame, timestamp) in enumerate(zip(frames, frame_times)):
            filename = output_path / f"frame_{timestamp:.2f}s.jpg"
            cv2.imwrite(str(filename), frame)  # pylint: disable=E1101
            saved_paths.append(str(filename))
            saved_timestamps.append(timestamp)

        return saved_paths, saved_timestamps


# Default arguments for testing
if __name__ == "__main__":
    load_dotenv()

    arguments = ActionArguments(
        name="video",
        transport="stdio",
        workspace=os.getenv("AWORLD_WORKSPACE", "~"),
    )

    try:
        service = VideoCollection(arguments)
        service.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
