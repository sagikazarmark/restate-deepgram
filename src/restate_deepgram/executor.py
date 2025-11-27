import logging
import tempfile
import typing
from pathlib import Path, PurePosixPath
from typing import Protocol

from deepgram import DeepgramClient
from deepgram.core import ApiError, RequestOptions
from deepgram.listen.v1.media.types.media_transcribe_request_callback_method import (
    MediaTranscribeRequestCallbackMethod,
)
from deepgram.listen.v1.media.types.media_transcribe_request_custom_intent_mode import (
    MediaTranscribeRequestCustomIntentMode,
)
from deepgram.listen.v1.media.types.media_transcribe_request_custom_topic_mode import (
    MediaTranscribeRequestCustomTopicMode,
)
from deepgram.listen.v1.media.types.media_transcribe_request_encoding import (
    MediaTranscribeRequestEncoding,
)
from deepgram.listen.v1.media.types.media_transcribe_request_model import (
    MediaTranscribeRequestModel,
)
from deepgram.listen.v1.media.types.media_transcribe_request_summarize import (
    MediaTranscribeRequestSummarize,
)
from deepgram.listen.v1.media.types.media_transcribe_request_version import (
    MediaTranscribeRequestVersion,
)
from deepgram.types.listen_v1response_metadata import ListenV1ResponseMetadata
from deepgram.types.listen_v1response_results import ListenV1ResponseResults
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, HttpUrl
from restate.exceptions import TerminalError

_logger = logging.getLogger(__name__)


class TranscribeRequestOutput(BaseModel):
    destination: AnyUrl | PurePosixPath | None = Field(
        default=None,
        description="The destination of the transcription file",
        union_mode="left_to_right",  # This is important to keep best match order (TODO: consider using a custom discriminator)
    )

    return_: bool | None = Field(
        alias="return",
        default=None,
        description="Whether to return the transcription",
    )


class TranscribeRequestOutputMixin:
    output: TranscribeRequestOutput = Field(
        default=TranscribeRequestOutput(),
        description="Output configuration",
    )


class TranscribeRequest(BaseModel):
    extra: str | typing.Sequence[str] | None = Field(
        default=None,
        description="Arbitrary key-value pairs that are attached to the API response for usage in downstream processing",
    )

    sentiment: bool | None = Field(
        default=None,
        description="Recognizes the sentiment throughout a transcript or text",
    )

    summarize: MediaTranscribeRequestSummarize | None = Field(
        default=None,
        description="Summarize content. For Listen API, supports string version option. For Read API, accepts boolean only.",
    )

    tag: str | typing.Sequence[str] | None = Field(
        default=None,
        description="Label your requests for the purpose of identification during usage reporting",
    )

    topics: bool | None = Field(
        default=None,
        description="Detect topics throughout a transcript or text",
    )

    custom_topic: str | typing.Sequence[str] | None = Field(
        default=None,
        description="Custom topics you want the model to detect within your input audio or text if present Submit up to 100.",
    )

    custom_topic_mode: MediaTranscribeRequestCustomTopicMode | None = Field(
        default=None,
        description="Sets how the model will interpret strings submitted to the custom_topic param. When strict, the model will only return topics submitted using the custom_topic param. When extended, the model will return its own detected topics in addition to those submitted using the custom_topic param",
    )

    intents: bool | None = Field(
        default=None,
        description="Recognizes speaker intent throughout a transcript or text",
    )

    custom_intent: str | typing.Sequence[str] | None = Field(
        default=None,
        description="Custom intents you want the model to detect within your input audio if present",
    )

    custom_intent_mode: MediaTranscribeRequestCustomIntentMode | None = Field(
        default=None,
        description="Sets how the model will interpret intents submitted to the custom_intent param. When strict, the model will only return intents submitted using the custom_intent param. When extended, the model will return its own detected intents in the custom_intent param.",
    )

    detect_entities: bool | None = Field(
        default=None,
        description="Identifies and extracts key entities from content in submitted audio",
    )

    detect_language: bool | None = Field(
        default=None,
        description="Identifies the dominant language spoken in submitted audio",
    )

    diarize: bool | None = Field(
        default=None,
        description="Recognize speaker changes. Each word in the transcript will be assigned a speaker number starting at 0",
    )

    dictation: bool | None = Field(
        default=None,
        description="Dictation mode for controlling formatting with dictated speech",
    )

    encoding: MediaTranscribeRequestEncoding | None = Field(
        default=None,
        description="Specify the expected encoding of your submitted audio",
    )

    filler_words: bool | None = Field(
        default=None,
        description="Filler Words can help transcribe interruptions in your audio, like 'uh' and 'um'",
    )

    keyterm: str | typing.Sequence[str] | None = Field(
        default=None,
        description="Key term prompting can boost or suppress specialized terminology and brands. Only compatible with Nova-3",
    )

    keywords: str | typing.Sequence[str] | None = Field(
        default=None,
        description="Keywords can boost or suppress specialized terminology and brands",
    )

    language: str | None = Field(
        default=None,
        description="The BCP-47 language tag that hints at the primary spoken language. Depending on the Model and API endpoint you choose only certain languages are available",
    )

    measurements: bool | None = Field(
        default=None,
        description="Spoken measurements will be converted to their corresponding abbreviations",
    )

    model: MediaTranscribeRequestModel | None = Field(
        default=None,
        description="AI model used to process submitted audio",
    )

    multichannel: bool | None = Field(
        default=None,
        description="Transcribe each audio channel independently",
    )

    numerals: bool | None = Field(
        default=None,
        description="Numerals converts numbers from written format to numerical format",
    )

    paragraphs: bool | None = Field(
        default=None,
        description="Splits audio into paragraphs to improve transcript readability",
    )

    profanity_filter: bool | None = Field(
        default=None,
        description="Profanity Filter looks for recognized profanity and converts it to the nearest recognized non-profane word or removes it from the transcript completely",
    )

    punctuate: bool | None = Field(
        default=None,
        description="Add punctuation and capitalization to the transcript",
    )

    redact: str | None = Field(
        default=None,
        description="Redaction removes sensitive information from your transcripts",
    )

    replace: str | typing.Sequence[str] | None = Field(
        default=None,
        description="Search for terms or phrases in submitted audio and replaces them",
    )

    search: str | typing.Sequence[str] | None = Field(
        default=None,
        description="Search for terms or phrases in submitted audio",
    )

    smart_format: bool | None = Field(
        default=None,
        description="Apply formatting to transcript output. When set to true, additional formatting will be applied to transcripts to improve readability",
    )

    utterances: bool | None = Field(
        default=None,
        description="Segments speech into meaningful semantic units",
    )

    utt_split: float | None = Field(
        default=None,
        description="Seconds to wait before detecting a pause between words in submitted audio",
    )

    version: MediaTranscribeRequestVersion | None = Field(
        default=None,
        description="Version of an AI model to use",
    )

    mip_opt_out: bool | None = Field(
        default=None,
        description="Opts out requests from the Deepgram Model Improvement Program. Refer to our Docs for pricing impacts before setting this to true. https://dpgr.am/deepgram-mip",
    )

    request_options: RequestOptions | None = Field(
        default=None,
        description="Request-specific configuration",
    )


class TranscribeAsyncRequest(BaseModel):
    callback: AnyUrl = Field(
        description="URL to which we'll make the callback request",
    )

    callback_method: MediaTranscribeRequestCallbackMethod | None = Field(
        default=None,
        description="HTTP method by which the callback request will be made",
    )


class TranscribeUrlRequest(TranscribeRequest, TranscribeRequestOutputMixin):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "url": "https://example.com/audio.wav",
                },
            ]
        }
    )

    url: HttpUrl = Field(description="URL of the audio or video file to transcribe")


class TranscribeUrlAsyncRequest(TranscribeUrlRequest, TranscribeAsyncRequest):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "url": "https://example.com/audio.wav",
                    "callback": "https://example.com/callback",
                },
            ]
        }
    )


class TranscribeFileRequest(TranscribeRequest, TranscribeRequestOutputMixin):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "input": "s3://bucket/audio.wav",
                },
            ]
        }
    )

    input: AnyUrl | PurePosixPath = Field(
        description="The audio file to transcribe",
        union_mode="left_to_right",  # This is important to keep best match order (TODO: consider using a custom discriminator)
    )


class TranscribeFileAsyncRequest(TranscribeFileRequest, TranscribeAsyncRequest):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "input": "s3://bucket/audio.wav",
                    "callback": "https://example.com/callback",
                },
            ]
        }
    )


class TranscribeResponse(BaseModel):
    metadata: ListenV1ResponseMetadata | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    results: ListenV1ResponseResults | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )


class TranscribeAsyncResponse(BaseModel):
    request_id: str


class Loader(typing.Protocol):
    def load(self, ref: AnyUrl | PurePosixPath, dst: Path): ...


class Persister(Protocol):
    def persist(
        self,
        ref: AnyUrl | PurePosixPath,
        src: bytes | bytearray | memoryview,
    ): ...


class Executor:
    def __init__(
        self,
        deepgram: DeepgramClient,
        loader: Loader,
        persister: Persister,
        logger: logging.Logger = _logger,
    ):
        self.deepgram = deepgram
        self.loader = loader
        self.persister = persister
        self.logger = logger

    def _handle_response(
        self,
        request: TranscribeRequestOutputMixin,
        response: BaseModel,
    ) -> bool:
        return_ = (
            bool(request.output.destination)
            if request.output.return_ is None
            else request.output.return_
        )

        if request.output.destination:
            self.persister.persist(
                request.output.destination,
                response.model_dump_json(indent=4).encode(),
            )

        return return_

    def transcribe_url(self, request: TranscribeUrlRequest) -> TranscribeResponse:
        params = request.model_dump(exclude_none=True, exclude={"output"})

        self.logger.info("Transcribing URL", extra={"url": request.url})

        try:
            apiResponse = self.deepgram.listen.v1.media.transcribe_url(**params)

            response = TranscribeResponse.model_validate(apiResponse.model_dump())

            return_ = self._handle_response(request, response)

            if return_:
                return response

            return TranscribeResponse()
        except ApiError as err:
            if _is_terminal(err):
                raise _convert_api_error(err) from err

            raise err

    def transcribe_url_async(
        self,
        request: TranscribeUrlAsyncRequest,
    ) -> TranscribeAsyncResponse:
        params = request.model_dump(exclude_none=True, exclude={"output"})

        self.logger.info("Transcribing URL", extra={"url": request.url})

        try:
            # callback is mandatory, so it always returns accepted response
            response = self.deepgram.listen.v1.media.transcribe_url(**params)

            return TranscribeAsyncResponse.model_validate(response.model_dump())
        except ApiError as err:
            if _is_terminal(err):
                raise _convert_api_error(err) from err

            raise err

    def transcribe_file(self, request: TranscribeFileRequest) -> TranscribeResponse:
        self.logger.info("Transcribing file", extra={"source": request.input})

        with tempfile.NamedTemporaryFile(delete=True) as file:
            self.loader.load(request.input, Path(file.name))

            params = request.model_dump(exclude_none=True, exclude={"input", "output"})

            try:
                apiResponse = self.deepgram.listen.v1.media.transcribe_file(
                    request=file,
                    **params,
                )

                response = TranscribeResponse.model_validate(apiResponse.model_dump())

                return_ = self._handle_response(request, response)

                if return_:
                    return response

                return TranscribeResponse()
            except ApiError as err:
                if _is_terminal(err):
                    raise _convert_api_error(err) from err

                raise err

    def transcribe_file_async(
        self, request: TranscribeFileAsyncRequest
    ) -> TranscribeAsyncResponse:
        self.logger.info("Transcribing file", extra={"source": request.input})

        with tempfile.NamedTemporaryFile(delete=True) as file:
            self.loader.load(request.input, Path(file.name))

            params = request.model_dump(exclude_none=True, exclude={"input", "output"})

            try:
                # callback is mandatory, so it always returns accepted response
                response = self.deepgram.listen.v1.media.transcribe_file(
                    request=file,
                    **params,
                )

                return TranscribeAsyncResponse.model_validate(response.model_dump())
            except ApiError as err:
                if _is_terminal(err):
                    raise _convert_api_error(err) from err

                raise err


def _is_terminal(err: ApiError) -> bool:
    # TODO: this should be better
    return err.status_code == 400


def _convert_api_error(err: ApiError) -> TerminalError:
    if isinstance(err.body, dict):
        msg = f"{err.body.get('err_msg', err.body)} (request id: {err.body.get('request_id', 'unknown')})"
    else:
        msg = f"{err.body}"

    return TerminalError(
        msg,
        status_code=err.status_code or 500,
    )
