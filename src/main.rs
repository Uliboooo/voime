use whisper_rs::{FullParams, WhisperContext};

fn main() {
    let current = std::env::current_dir().unwrap();

    let model_path = current.join("resource").join("ggml-small-q8_0.bin");
    let audio_path = current.join("resource").join("test.wav");

    let ctx =
        WhisperContext::new_with_params(model_path.to_str().unwrap(), Default::default()).unwrap();
    let mut reader = hound::WavReader::open(audio_path).unwrap();
    let audio_data = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect::<Vec<f32>>();

    let mut parms = FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 0 });
    parms.set_language(Some("ja"));

    let mut state = ctx.create_state().unwrap();
    state.full(parms, &audio_data).unwrap();

    // let num_seg = state.full_n_segments();

    let text = state.as_iter().map(|f| format!("{f}")).collect::<String>();
    println!("{text}");
}
