//! End-to-end pipeline integration tests using mock providers.

use bytes::Bytes;
use herta_core::{
    Message, Role,
    audio::{AudioFormat, Utterance},
    memory::Memory,
    mocks::{EchoLlm, FixedStt, InMemoryMemory, RecordingAudioOutput, SilentTts},
    pipeline::{PipelineConfig, VoicePipeline},
};
use std::sync::Arc;

fn test_utterance(ms: u64) -> Utterance {
    Utterance {
        pcm: Bytes::from(vec![0u8; 16_000 * 4]),
        format: AudioFormat::default(),
        duration_ms: ms,
    }
}

#[tokio::test]
async fn end_to_end_text_turn() {
    let memory = Arc::new(InMemoryMemory::default());
    let output = Arc::new(RecordingAudioOutput::default());
    let pipeline = VoicePipeline::builder()
        .llm(Arc::new(EchoLlm::new("mock")))
        .memory(memory.clone())
        .tts(Arc::new(SilentTts))
        .output(output.clone())
        .config(PipelineConfig {
            context_messages: 4,
            locked_prefix: vec![Message::system("persona")],
            tenant_id: Some("unit-test".into()),
        })
        .build()
        .unwrap();

    let reply = pipeline.run_text_turn("hello", None).await.unwrap();
    assert_eq!(reply, "echo: hello");
    let history = memory.load_context(10).await.unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].role, Role::User);
    assert_eq!(history[1].role, Role::Assistant);
    assert_eq!(output.recorded().len(), 1);
}

#[tokio::test]
async fn end_to_end_voice_turn() {
    let memory = Arc::new(InMemoryMemory::default());
    let output = Arc::new(RecordingAudioOutput::default());
    let pipeline = VoicePipeline::builder()
        .llm(Arc::new(EchoLlm::new("mock")))
        .memory(memory.clone())
        .stt(Arc::new(FixedStt::new("привет")))
        .tts(Arc::new(SilentTts))
        .output(output.clone())
        .build()
        .unwrap();

    let reply = pipeline
        .run_voice_turn(&test_utterance(1_000), None)
        .await
        .unwrap();
    assert_eq!(reply.as_deref(), Some("echo: привет"));
    assert_eq!(memory.load_context(10).await.unwrap().len(), 2);
    assert_eq!(output.recorded().len(), 1);
}

#[tokio::test]
async fn empty_transcript_does_not_trigger_llm() {
    let memory = Arc::new(InMemoryMemory::default());
    let pipeline = VoicePipeline::builder()
        .llm(Arc::new(EchoLlm::new("mock")))
        .memory(memory.clone())
        .stt(Arc::new(FixedStt::new("")))
        .build()
        .unwrap();

    let reply = pipeline
        .run_voice_turn(&test_utterance(500), None)
        .await
        .unwrap();
    assert!(reply.is_none());
    assert!(memory.load_context(10).await.unwrap().is_empty());
}

#[tokio::test]
async fn preview_context_reflects_stored_history() {
    let memory = Arc::new(InMemoryMemory::default());
    memory.append_turn("previous-user", "previous-assistant").await.unwrap();

    let pipeline = VoicePipeline::builder()
        .llm(Arc::new(EchoLlm::new("mock")))
        .memory(memory.clone())
        .config(PipelineConfig {
            context_messages: 10,
            locked_prefix: vec![Message::system("persona")],
            tenant_id: None,
        })
        .build()
        .unwrap();

    let preview = pipeline.preview_context("next").await.unwrap();
    assert!(
        preview
            .iter()
            .any(|m| m.content == "previous-user" && m.role == Role::User)
    );
    assert_eq!(preview.last().unwrap().content, "next");
}
