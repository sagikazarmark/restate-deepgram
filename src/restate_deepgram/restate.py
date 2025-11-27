import restate

from .executor import (
    Executor,
    TranscribeAsyncResponse,
    TranscribeFileAsyncRequest,
    TranscribeFileRequest,
    TranscribeResponse,
    TranscribeUrlAsyncRequest,
    TranscribeUrlRequest,
)


def create_service(
    downloader: Executor,
    service_name: str = "Deepgram",
) -> restate.Service:
    service = restate.Service(service_name)

    register_service(downloader, service)

    return service


def register_service(
    executor: Executor,
    service: restate.Service,
):
    @service.handler("transcribeUrl")
    async def transcribe_url(
        ctx: restate.Context,
        request: TranscribeUrlRequest,
    ) -> TranscribeResponse:
        return await ctx.run_typed(
            "transcribe_url",
            executor.transcribe_url,
            request=request,
        )

    @service.handler("transcribeUrlAsync")
    async def transcribe_url_async(
        ctx: restate.Context,
        request: TranscribeUrlAsyncRequest,
    ) -> TranscribeAsyncResponse:
        return await ctx.run_typed(
            "transcribe_url_async",
            executor.transcribe_url_async,
            request=request,
        )

    @service.handler("transcribeFile")
    async def transcribe_file(
        ctx: restate.Context,
        request: TranscribeFileRequest,
    ) -> TranscribeResponse:
        return await ctx.run_typed(
            "transcribe_file",
            executor.transcribe_file,
            request=request,
        )

    @service.handler("transcribeFileAsync")
    async def transcribe_file_async(
        ctx: restate.Context,
        request: TranscribeFileAsyncRequest,
    ) -> TranscribeAsyncResponse:
        return await ctx.run_typed(
            "transcribe_file_async",
            executor.transcribe_file_async,
            request=request,
        )
