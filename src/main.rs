use arboard::Clipboard;
use core::panic;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample};
use enigo::{Enigo, Key, Keyboard};
use std::fs::File;
use std::io::BufWriter;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};
use uinput::event::{Press, Release};
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

#[derive(Debug)]
pub struct Parms<'a> {
    lang: Option<&'a str>,
    thread: ThreadParm,
}

#[derive(Debug)]
pub enum ThreadParm {
    Max,
    Half,
    Quarter,
    CustomDiv(usize),
    Custom(usize),
}

fn transcribe<T: AsRef<Path>>(model_path: T, audio_path: T, in_parms: &Parms) -> String {
    let ctx =
        WhisperContext::new_with_params(model_path.as_ref().to_str().unwrap(), Default::default())
            .unwrap();

    let mut reader = hound::WavReader::open(audio_path).unwrap();
    let spec = reader.spec();
    // println!("File Spec: {}Hz, {}ch", spec.sample_rate, spec.channels);

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
    let num_trheads = std::thread::available_parallelism()
        .unwrap_or(NonZeroUsize::new(1).unwrap())
        .get();
    let n = match in_parms.thread {
        ThreadParm::Max => num_trheads,
        ThreadParm::Half => num_trheads / 2,
        ThreadParm::Quarter => num_trheads / 4,
        ThreadParm::Custom(i) => {
            if i > num_trheads {
                num_trheads
            } else {
                i
            }
        }
        ThreadParm::CustomDiv(d) => {
            if d > 0 && d < num_trheads {
                num_trheads / d
            } else {
                1
            }
        }
    } as std::ffi::c_int;
    parms.set_n_threads(n);

    parms.set_language(in_parms.lang);

    let mut state = ctx.create_state().unwrap();
    state.full(parms, &audio_data).unwrap();

    // let num_seg = state.full_n_segments();

    state.as_iter().map(|f| format!("{f}")).collect::<String>()
}

fn send_text<T: AsRef<str>>(text: T) {
    let _ = std::process::Command::new("wl-copy")
        .arg(text.as_ref())
        .status();

    std::thread::sleep(std::time::Duration::from_millis(100));

    let _ = std::process::Command::new("wtype")
        .args(["-M", "ctrl", "v", "-m", "ctrl"])
        .status();

    // let mut clipb = Clipboard::new().unwrap();
    // clipb.set_text(text.as_ref()).unwrap();
    //
    // let mut enigo = Enigo::new(&enigo::Settings::default()).unwrap();
    // std::thread::sleep(std::time::Duration::from_millis(500));
    //
    // enigo.key(Key::LControl, enigo::Direction::Press).unwrap();
    // enigo
    //     .key(Key::Unicode('v'), enigo::Direction::Click)
    //     .unwrap();
    // enigo.key(Key::LControl, enigo::Direction::Release).unwrap();

    // let mut device = uinput::default()
    //     .unwrap()
    //     .name("virtual voice input")
    //     .unwrap()
    //     .event(uinput::event::Keyboard::All)
    //     .unwrap()
    //     .create()
    //     .unwrap();
    // std::thread::sleep(std::time::Duration::from_secs(1));
    //
    // // 3. Ctrl + V の送信
    // // LeftControlをプレス状態にする
    // device
    //     .press(&uinput::event::keyboard::Key::LeftControl)
    //     .unwrap();
    // device.synchronize().unwrap();
    // std::thread::sleep(std::time::Duration::from_millis(10));
    //
    // // Vをカチッと押す
    // device.click(&uinput::event::keyboard::Key::V).unwrap();
    // device.synchronize().unwrap();
    // std::thread::sleep(std::time::Duration::from_millis(10));
    //
    // // LeftControlを離す
    // device
    //     .release(&uinput::event::keyboard::Key::LeftControl)
    //     .unwrap();
    // device.synchronize().unwrap();

    println!("pasted.");
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

    let parms = Parms {
        lang: Some("ja"),
        thread: ThreadParm::Half,
    };
    let text = transcribe(model_path, test_path, &parms);
    // record(&save_path);
    // let text = transcribe(model_path, save_path, parms);

    send_text(text);
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use whisper_rs::install_logging_hooks;

    use crate::{Parms, ThreadParm, send_text, transcribe};

    #[test]
    fn test_ttrb() {
        install_logging_hooks();
        let list = [
            "tiny-q5_1",
            "tiny-q8_0",
            "base-q5_1",
            "base-q8_0",
            "small-q5_1",
            "small-q8_0",
        ];
        let work_path = std::env::current_dir().unwrap().join("resource");
        let test_path = work_path.join("test.wav");

        let mut res_buf = String::new();
        let parms = Parms {
            lang: Some("ja"),
            thread: ThreadParm::Half,
        };

        for i in 1..11 {
            // println!("the {} time", i);
            for model_name in list {
                let model_path = work_path.join(format!("ggml-{}.bin", model_name));
                let start_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap();
                let result = transcribe(model_path, test_path.clone(), &parms);
                let end_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap();
                let time = (end_time - start_time).as_millis();

                res_buf.push_str(&format!("{i},{model_name},{time},{result}\n"));
            }
        }

        println!("index,model,time,result");
        println!("{res_buf}");
    }

    #[test]
    fn test_once() {
        install_logging_hooks();
        let parms = Parms {
            lang: Some("ja"),
            thread: ThreadParm::Half,
        };

        let work_path = std::env::current_dir().unwrap().join("resource");
        let model_path = work_path.join("ggml-small-q8_0.bin");
        let test_path = work_path.join("test.wav");

        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let result = transcribe(model_path, test_path.clone(), &parms);
        let end_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let time = (end_time - start_time).as_millis();

        println!("small-q8_0,{time},{result}\n");
    }

    fn new_parms(th: ThreadParm) -> Parms<'static> {
        Parms {
            lang: Some("ja"),
            thread: th,
        }
    }

    #[test]
    fn benchmark() {
        install_logging_hooks();
        let list = [
            "tiny-q5_1",
            "tiny-q8_0",
            "base-q5_1",
            "base-q8_0",
            "small-q5_1",
            "small-q8_0",
        ];
        let work_path = std::env::current_dir().unwrap().join("resource");
        let test_path = work_path.join("test.wav");

        let mut res_buf = String::new();

        let parms = [
            new_parms(ThreadParm::Max),
            new_parms(ThreadParm::Half),
            new_parms(ThreadParm::Quarter),
        ];

        let num_trheads = std::thread::available_parallelism()
            .unwrap_or(NonZeroUsize::new(1).unwrap())
            .get();

        for p in parms {
            for model_name in list {
                let n = match &p.thread {
                    ThreadParm::Max => num_trheads,
                    ThreadParm::Half => num_trheads / 2,
                    ThreadParm::Quarter => num_trheads / 4,
                    ThreadParm::Custom(i) => {
                        if *i > num_trheads {
                            num_trheads
                        } else {
                            *i
                        }
                    }
                    ThreadParm::CustomDiv(d) => {
                        if *d > 0 && *d < num_trheads {
                            num_trheads / d
                        } else {
                            1
                        }
                    }
                } as std::ffi::c_int;

                let model_path = work_path.join(format!("ggml-{}.bin", model_name));
                for i in 1..4 {
                    let start_time = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap();
                    let result = transcribe(&model_path, &test_path.clone(), &p);

                    let end_time = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap();
                    let time = (end_time - start_time).as_millis();

                    res_buf.push_str(&format!(
                        "{:?},{n},{i},{model_name},{time},{result}\n",
                        p.thread
                    ));
                }
            }
        }

        println!("\nprams-mode,threads,index,model,time,result\n{res_buf}");
    }

    #[test]
    fn send_test() {
        let t = "Hellow world!";
        send_text(t);
    }
}
