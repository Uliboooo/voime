use core::panic;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::{Arc, Mutex};
use whisper_rs::{FullParams, WhisperContext, install_logging_hooks};

fn record<T: AsRef<Path>>(save_path: T) {
    let host = cpal::host_from_id(
        cpal::available_hosts()
            .into_iter()
            .find(|id| *id == cpal::HostId::Alsa)
            .unwrap(),
    )
    .unwrap();

    let device = host.default_input_device().unwrap();

    let config = if device.supports_input() {
        device.default_input_config()
    } else {
        device.default_output_config()
    }
    .unwrap();

    if !save_path.as_ref().exists() {
        std::fs::File::create(&save_path).unwrap();
    }

    let spec = wav_spec_from_config(&config);
    let writer = hound::WavWriter::create(save_path, spec).unwrap();
    let writer = Arc::new(Mutex::new(Some(writer)));

    let writer_2 = writer.clone();

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {err}");
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::I8 => device
            .build_input_stream(
                &config.into(),
                move |data, _: &_| write_input_data::<i8, i8>(data, &writer_2),
                err_fn,
                None,
            )
            .unwrap(),
        cpal::SampleFormat::I16 => device
            .build_input_stream(
                &config.into(),
                move |data, _: &_| write_input_data::<i16, i16>(data, &writer_2),
                err_fn,
                None,
            )
            .unwrap(),
        cpal::SampleFormat::I32 => device
            .build_input_stream(
                &config.into(),
                move |data, _: &_| write_input_data::<i32, i32>(data, &writer_2),
                err_fn,
                None,
            )
            .unwrap(),
        cpal::SampleFormat::F32 => device
            .build_input_stream(
                &config.into(),
                move |data, _: &_| write_input_data::<f32, f32>(data, &writer_2),
                err_fn,
                None,
            )
            .unwrap(),
        sample_format => {
            panic!("Unsupported sample format '{sample_format}'");
        }
    };

    println!("start recording...");
    stream.play().unwrap();

    // timing by user button
    std::thread::sleep(std::time::Duration::from_secs(13));
    drop(stream);
    writer.lock().unwrap().take().unwrap().finalize().unwrap();
    println!("stopped.");
}

fn sample_format(format: cpal::SampleFormat) -> hound::SampleFormat {
    if format.is_float() {
        hound::SampleFormat::Float
    } else if format.is_int() {
        hound::SampleFormat::Int
    } else {
        panic!("DSD formats cannot be written to WAV files");
    }
}

fn wav_spec_from_config(config: &cpal::SupportedStreamConfig) -> hound::WavSpec {
    hound::WavSpec {
        channels: config.channels() as _,
        sample_rate: config.sample_rate() as _,
        bits_per_sample: (config.sample_format().sample_size() * 8) as _,
        sample_format: sample_format(config.sample_format()),
    }
}

type WavWriterHandle = Arc<Mutex<Option<hound::WavWriter<BufWriter<File>>>>>;

fn write_input_data<T, U>(input: &[T], writer: &WavWriterHandle)
where
    T: Sample,
    U: Sample + hound::Sample + FromSample<T>,
{
    let Ok(mut guard) = writer.lock() else {
        return;
    };
    if let Some(writer) = guard.as_mut() {
        for &sample in input.iter() {
            let sample: U = U::from_sample(sample);
            writer.write_sample(sample).ok();
        }
    }
}

fn transcribe<T: AsRef<Path>>(model_path: T, audio_path: T) -> String {
    let ctx =
        WhisperContext::new_with_params(model_path.as_ref().to_str().unwrap(), Default::default())
            .unwrap();

    let mut reader = hound::WavReader::open(audio_path).unwrap();
    let spec = reader.spec();
    println!("File Spec: {}Hz, {}ch", spec.sample_rate, spec.channels);

    let src_samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
        reader.samples::<f32>().map(|s| s.unwrap()).collect()
    } else {
        reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect()
    };

    let mono_samples = if spec.channels > 1 {
        src_samples
            .chunks_exact(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    } else {
        src_samples
    };

    let audio_data = if spec.sample_rate != 16000 {
        let factor = spec.sample_rate as f64 / 16000.0;
        let target_len = (mono_samples.len() as f64 / factor) as usize;
        let mut result = Vec::with_capacity(target_len);
        for i in 0..target_len {
            let pos = i as f64 * factor;
            let idx = pos as usize;
            if idx + 1 < mono_samples.len() {
                let frac = pos - idx as f64;
                result.push(
                    (mono_samples[idx] as f64 * (1.0 - frac) + mono_samples[idx + 1] as f64 * frac)
                        as f32,
                );
            } else {
                result.push(mono_samples[idx]);
            }
        }
        result
    } else {
        mono_samples
    };

    let mut parms = FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 0 });
    parms.set_language(Some("ja"));

    let mut state = ctx.create_state().unwrap();
    state.full(parms, &audio_data).unwrap();

    // let num_seg = state.full_n_segments();

    state.as_iter().map(|f| format!("{f}")).collect::<String>()
}

fn send_text<T: AsRef<str>>(text: T) {
    std::process::Command::new("wtype")
        .arg(text.as_ref())
        .output()
        .unwrap();
}

fn main() {
    // if release mode, hide whisper logs
    if cfg!(not(debug_assertions)) {
        install_logging_hooks();
    }

    // TODO: PATH will be written in config.
    let current = std::env::current_dir().unwrap();

    let model_path = current.join("resource").join("ggml-small-q8_0.bin");
    let test_path = current.join("resource").join("test.wav");
    let save_path = current.join("resource").join("save.wav");

    // let text = transcribe(model_path, audio_path);
    record(&save_path);
    let text = transcribe(model_path, save_path);

    println!("{text}");
    // send_text(text);
}
