use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, Stream};
use directories::ProjectDirs;
use gtk4::prelude::*;
use gtk4::{Application, ApplicationWindow, Box, Button, CssProvider, EventControllerKey, Label, Orientation};
use gtk4_layer_shell::{Edge, Layer, LayerShell};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use whisper_rs::{FullParams, WhisperContext, install_logging_hooks};

const DEFAULT_MODEL_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin";

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AppConfig {
    model_path: Option<PathBuf>,
    language: String,
    dark_mode: bool,
    opacity: f32,
    auto_copy: bool,
    auto_start_record: bool,
    key_record: Vec<String>,
    key_copy: Vec<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            language: "ja".to_string(),
            dark_mode: true,
            opacity: 0.9,
            auto_copy: true,
            auto_start_record: false,
            key_record: vec!["q".to_string(), "Return".to_string()],
            key_copy: vec!["c".to_string(), "y".to_string(), "space".to_string()],
        }
    }
}

#[derive(Debug, Clone)]
enum AppMsg {
    TranscriptionDone(String),
    Error(String),
    DownloadProgress(f64),
    ModelReady(PathBuf),
}

fn get_project_dirs() -> ProjectDirs {
    ProjectDirs::from("com", "github", "voime").expect("Failed to get project directories")
}

fn load_config() -> AppConfig {
    let proj_dirs = get_project_dirs();
    let config_dir = proj_dirs.config_dir();
    let config_path = config_dir.join("config.toml");

    if config_path.exists() {
        let content = fs::read_to_string(&config_path).unwrap_or_default();
        toml::from_str(&content).unwrap_or_else(|_| {
            let config = AppConfig::default();
            let _ = fs::write(&config_path, toml::to_string(&config).unwrap_or_default());
            config
        })
    } else {
        let config = AppConfig::default();
        fs::create_dir_all(config_dir).ok();
        let content = toml::to_string(&config).unwrap_or_default();
        fs::write(&config_path, content).ok();
        config
    }
}

fn get_wav_spec(config: &cpal::SupportedStreamConfig) -> hound::WavSpec {
    hound::WavSpec {
        channels: config.channels() as _,
        sample_rate: config.sample_rate() as _,
        bits_per_sample: (config.sample_format().sample_size() * 8) as _,
        sample_format: if config.sample_format().is_float() {
            hound::SampleFormat::Float
        } else {
            hound::SampleFormat::Int
        },
    }
}

type WavWriterHandle = Arc<Mutex<Option<hound::WavWriter<BufWriter<File>>>>>;

fn write_input_data<T, U>(input: &[T], writer: &WavWriterHandle)
where
    T: Sample,
    U: Sample + hound::Sample + FromSample<T>,
{
    if let Ok(mut guard) = writer.lock() {
        if let Some(writer) = guard.as_mut() {
            for &sample in input.iter() {
                let sample: U = U::from_sample(sample);
                writer.write_sample(sample).ok();
            }
        }
    }
}

fn start_recording(save_path: PathBuf) -> (Stream, WavWriterHandle) {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device available");
    let config = device.default_input_config().expect("Failed to get default input config");

    let spec = get_wav_spec(&config);
    let writer = hound::WavWriter::create(save_path, spec).expect("Failed to create WavWriter");
    let writer = Arc::new(Mutex::new(Some(writer)));
    let writer_2 = writer.clone();

    let err_fn = |err| eprintln!("an error occurred on stream: {err}");

    let stream = match config.sample_format() {
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| write_input_data::<i16, i16>(data, &writer_2),
            err_fn,
            None,
        ),
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| write_input_data::<f32, f32>(data, &writer_2),
            err_fn,
            None,
        ),
        _ => panic!("Unsupported sample format"),
    }.expect("Failed to build input stream");

    stream.play().expect("Failed to start stream");
    (stream, writer)
}

pub struct TranscribeParams<'a> {
    lang: Option<&'a str>,
    threads: usize,
}

fn transcribe(model_path: &Path, audio_path: &Path, params: &TranscribeParams) -> Result<String, String> {
    let ctx = WhisperContext::new_with_params(model_path.to_str().ok_or("Invalid model path")?, Default::default())
        .map_err(|e| format!("Failed to create WhisperContext: {e}"))?;

    let mut reader = hound::WavReader::open(audio_path).map_err(|e| format!("Failed to open audio: {e}"))?;
    let spec = reader.spec();

    let src_samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
        reader.samples::<f32>().map(|s| s.unwrap()).collect()
    } else {
        reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
    };

    let mono_samples = if spec.channels > 1 {
        src_samples.chunks_exact(spec.channels as usize).map(|c| c.iter().sum::<f32>() / spec.channels as f32).collect()
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
                result.push((mono_samples[idx] as f64 * (1.0 - frac) + mono_samples[idx+1] as f64 * frac) as f32);
            } else if idx < mono_samples.len() {
                result.push(mono_samples[idx]);
            }
        }
        result
    } else {
        mono_samples
    };

    let mut whisper_params = FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 0 });
    whisper_params.set_n_threads(params.threads as i32);
    whisper_params.set_language(params.lang);

    let mut state = ctx.create_state().map_err(|e| format!("Failed to create state: {e}"))?;
    state.full(whisper_params, &audio_data).map_err(|e| format!("Transcription failed: {e}"))?;

    Ok(state.as_iter().map(|f| format!("{f}")).collect::<String>())
}

fn main() {
    if cfg!(not(debug_assertions)) {
        install_logging_hooks();
    }

    let app = Application::builder()
        .application_id("com.github.coyuki.voime")
        .build();

    app.connect_activate(build_ui);
    app.run();
}

fn build_ui(app: &Application) {
    let config = load_config();

    if config.dark_mode {
        if let Some(settings) = gtk4::Settings::default() {
            settings.set_gtk_application_prefer_dark_theme(true);
        }
    }

    let window = ApplicationWindow::builder()
        .application(app)
        .title("Voime")
        .default_width(600)
        .default_height(100)
        .build();

    if config.opacity < 1.0 {
        let provider = CssProvider::new();
        provider.load_from_data(&format!("
            window {{ 
                background-color: transparent; 
            }}
            .main-box {{ 
                background-color: rgba(30, 30, 30, {}); 
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
        ", config.opacity));
        gtk4::style_context_add_provider_for_display(
            &WidgetExt::display(&window),
            &provider,
            gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }

    window.init_layer_shell();
    window.set_layer(Layer::Overlay);
    window.set_anchor(Edge::Top, true);
    window.set_margin(Edge::Top, 100);
    window.set_keyboard_mode(gtk4_layer_shell::KeyboardMode::OnDemand);

    let main_box = Box::builder()
        .orientation(Orientation::Vertical)
        .spacing(10)
        .margin_top(20)
        .margin_bottom(20)
        .margin_start(20)
        .margin_end(20)
        .build();
    
    if config.opacity < 1.0 {
        main_box.add_css_class("main-box");
    }

    let status_label = Label::new(Some("Initializing..."));
    let action_button = Button::with_label("Wait...");
    action_button.set_sensitive(false);

    let result_box = Box::builder()
        .orientation(Orientation::Horizontal)
        .spacing(10)
        .visible(false)
        .build();
    let result_label = Label::builder()
        .label("")
        .hexpand(true)
        .xalign(0.0)
        .wrap(true)
        .build();
    let copy_button = Button::with_label("Copy");
    copy_button.set_sensitive(false);

    result_box.append(&result_label);
    result_box.append(&copy_button);

    main_box.append(&status_label);
    main_box.append(&action_button);
    main_box.append(&result_box);

    window.set_child(Some(&main_box));

    let recording_state = Arc::new(Mutex::new(None::<(Stream, WavWriterHandle)>));
    let current_text = Arc::new(Mutex::new(String::new()));
    let (tx, rx) = std::sync::mpsc::channel::<AppMsg>();
    let model_ready_path = Arc::new(Mutex::new(None::<PathBuf>));

    let proj_dirs = get_project_dirs();
    let data_dir = proj_dirs.data_dir();
    let cache_dir = proj_dirs.cache_dir();
    fs::create_dir_all(data_dir).ok();
    fs::create_dir_all(cache_dir).ok();

    let save_path = cache_dir.join("save.wav");
    let default_model_path = data_dir.join("ggml-small-q8_0.bin");

    let action_button_weak = action_button.downgrade();
    let status_label_weak = status_label.downgrade();
    let result_box_weak = result_box.downgrade();
    let result_label_weak = result_label.downgrade();
    let copy_button_weak = copy_button.downgrade();
    let current_text_clone = current_text.clone();
    let model_ready_path_clone = model_ready_path.clone();
    let window_weak = window.downgrade();
    let auto_copy_enabled = config.auto_copy;
    let auto_start_record = config.auto_start_record;

    let toggle_recording = {
        let recording_state = recording_state.clone();
        let save_path = save_path.clone();
        let model_ready_path = model_ready_path.clone();
        let tx = tx.clone();
        let action_button_weak = action_button.downgrade();
        let status_label_weak = status_label.downgrade();
        let result_box_weak = result_box.downgrade();
        let language = config.language.clone();

        move || {
            let action_button = if let Some(b) = action_button_weak.upgrade() { b } else { return; };
            if !action_button.is_sensitive() { return; }

            let model_path = if let Some(p) = model_ready_path.lock().unwrap().clone() { p } else { return; };

            let mut state = recording_state.lock().unwrap();
            let status_label = status_label_weak.upgrade().unwrap();
            let result_box = result_box_weak.upgrade().unwrap();

            if state.is_none() {
                let (stream, writer) = start_recording(save_path.clone());
                *state = Some((stream, writer));
                action_button.set_label("Stop");
                status_label.set_text("Recording...");
                result_box.set_visible(false);
            } else {
                if let Some((stream, writer)) = state.take() {
                    drop(stream);
                    if let Ok(mut guard) = writer.lock() {
                        if let Some(w) = guard.take() { w.finalize().ok(); }
                    }
                }
                action_button.set_sensitive(false);
                action_button.set_label("Processing...");
                status_label.set_text("Processing...");

                let tx_clone = tx.clone();
                let save_path_clone = save_path.clone();
                let lang_clone = language.clone();
                std::thread::spawn(move || {
                    let threads = std::thread::available_parallelism().map(|n| n.get() / 2).unwrap_or(1);
                    let params = TranscribeParams { lang: Some(&lang_clone), threads };
                    match transcribe(&model_path, &save_path_clone, &params) {
                        Ok(text) => { let _ = tx_clone.send(AppMsg::TranscriptionDone(text)); },
                        Err(e) => { let _ = tx_clone.send(AppMsg::Error(e)); },
                    };
                });
            }
        }
    };

    let toggle_recording_clone = Arc::new(toggle_recording);
    let tr_1 = toggle_recording_clone.clone();

    glib::idle_add_local(move || {
        while let Ok(msg) = rx.try_recv() {
            match msg {
                AppMsg::ModelReady(path) => {
                    if let (Some(btn), Some(lbl)) = (action_button_weak.upgrade(), status_label_weak.upgrade()) {
                        btn.set_sensitive(true);
                        btn.set_label("Record");
                        lbl.set_text("Ready (Press key to record)");
                        *model_ready_path_clone.lock().unwrap() = Some(path);

                        if auto_start_record {
                            tr_1();
                        }
                    }
                }
                AppMsg::DownloadProgress(p) => {
                    if let Some(lbl) = status_label_weak.upgrade() {
                        lbl.set_text(&format!("Downloading Model: {:.1}%", p * 100.0));
                    }
                }
                AppMsg::TranscriptionDone(text) => {
                    if let (Some(btn), Some(lbl), Some(res_box), Some(res_lbl), Some(cp_btn)) = (
                        action_button_weak.upgrade(),
                        status_label_weak.upgrade(),
                        result_box_weak.upgrade(),
                        result_label_weak.upgrade(),
                        copy_button_weak.upgrade(),
                    ) {
                        btn.set_sensitive(true);
                        btn.set_label("Record Again");
                        lbl.set_text("Done");
                        res_lbl.set_text(&text);
                        res_box.set_visible(true);
                        cp_btn.set_sensitive(true);
                        *current_text_clone.lock().unwrap() = text.clone();

                        if auto_copy_enabled {
                            if let Some(window) = window_weak.upgrade() {
                                let display = WidgetExt::display(&window);
                                let clipboard = display.clipboard();
                                clipboard.set_text(&text);
                            }
                        }
                    }
                }
                AppMsg::Error(e) => {
                    if let (Some(btn), Some(lbl)) = (action_button_weak.upgrade(), status_label_weak.upgrade()) {
                        btn.set_sensitive(true);
                        lbl.set_text(&format!("Error: {}", e));
                    }
                }
            }
        }
        glib::ControlFlow::Continue
    });

    // Handle Model Check & Download
    let tx_dl = tx.clone();
    let config_model_path = config.model_path.clone();
    let default_model_path_clone = default_model_path.clone();
    std::thread::spawn(move || {
        let path = config_model_path.unwrap_or(default_model_path_clone);
        if path.exists() {
            let _ = tx_dl.send(AppMsg::ModelReady(path));
        } else {
            // Download
            let mut response = reqwest::blocking::get(DEFAULT_MODEL_URL).expect("Failed to download model");
            let total_size = response.content_length().unwrap_or(1);
            let mut file = File::create(&path).expect("Failed to create model file");
            let mut buffer = [0; 8192];
            let mut downloaded = 0;

            while let Ok(n) = response.read(&mut buffer) {
                if n == 0 { break; }
                file.write_all(&buffer[..n]).ok();
                downloaded += n as u64;
                let _ = tx_dl.send(AppMsg::DownloadProgress(downloaded as f64 / total_size as f64));
            }
            let _ = tx_dl.send(AppMsg::ModelReady(path));
        }
    });

    let tr_click = toggle_recording_clone.clone();
    action_button.connect_clicked(move |_| { tr_click(); });

    let copy_to_clipboard = {
        let current_text = current_text.clone();
        let window_weak = window.downgrade();
        move || {
            let text = current_text.lock().unwrap();
            if !text.is_empty() {
                if let Some(window) = window_weak.upgrade() {
                    let display = WidgetExt::display(&window);
                    let clipboard = display.clipboard();
                    clipboard.set_text(&text);
                }
            }
        }
    };

    let copy_to_clipboard_clone = Arc::new(copy_to_clipboard);
    let cc_1 = copy_to_clipboard_clone.clone();
    copy_button.connect_clicked(move |_| { cc_1(); });

    let key_controller = EventControllerKey::new();
    let tr_2 = toggle_recording_clone.clone();
    let cc_2 = copy_to_clipboard_clone.clone();
    let config_keys = config.clone();
    let app_weak = app.downgrade();

    key_controller.connect_key_pressed(move |_controller, key, _keycode, _state| {
        let key_name = key.name().unwrap_or_default();
        if config_keys.key_record.contains(&key_name.to_string()) {
            tr_2();
            glib::Propagation::Stop
        } else if config_keys.key_copy.contains(&key_name.to_string()) {
            cc_2();
            glib::Propagation::Stop
        } else if key_name == "Escape" {
            if let Some(app) = app_weak.upgrade() {
                app.quit();
            }
            glib::Propagation::Stop
        } else {
            glib::Propagation::Proceed
        }
    });
    window.add_controller(key_controller);

    window.present();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;
    use arboard::Clipboard;

    #[derive(Debug, Clone, Copy)]
    pub enum ThreadParm { Max, Half, Quarter }

    fn get_thread_count(p: ThreadParm) -> usize {
        let num_threads = std::thread::available_parallelism()
            .unwrap_or(NonZeroUsize::new(1).unwrap())
            .get();
        match p {
            ThreadParm::Max => num_threads,
            ThreadParm::Half => num_threads / 2,
            ThreadParm::Quarter => num_threads / 4,
        }
    }

    #[test]
    fn test_ttrb() {
        if cfg!(not(debug_assertions)) { install_logging_hooks(); }
        let list = ["tiny-q5_1", "tiny-q8_0", "base-q5_1", "base-q8_0", "small-q5_1", "small-q8_0"];
        let work_path = std::env::current_dir().unwrap().join("resource");
        let test_path = work_path.join("test.wav");
        let mut res_buf = String::new();
        let threads = get_thread_count(ThreadParm::Half);

        for i in 1..11 {
            for model_name in list {
                let model_path = work_path.join(format!("ggml-{}.bin", model_name));
                let start_time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
                let params = TranscribeParams { lang: Some("ja"), threads };
                let result = transcribe(&model_path, &test_path, &params).unwrap_or_else(|e| e);
                let end_time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
                let time = (end_time - start_time).as_millis();
                res_buf.push_str(&format!("{i},{model_name},{time},{result}\n"));
            }
        }
        println!("index,model,time,result\n{res_buf}");
    }

    #[test]
    fn test_once() {
        if cfg!(not(debug_assertions)) { install_logging_hooks(); }
        let threads = get_thread_count(ThreadParm::Half);
        let params = TranscribeParams { lang: Some("ja"), threads };
        let work_path = std::env::current_dir().unwrap().join("resource");
        let model_path = work_path.join("ggml-small-q8_0.bin");
        let test_path = work_path.join("test.wav");
        let start_time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
        let result = transcribe(&model_path, &test_path, &params).expect("Transcription failed");
        let end_time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
        let time = (end_time - start_time).as_millis();
        println!("small-q8_0,{time},{result}\n");
    }

    #[test]
    fn benchmark() {
        if cfg!(not(debug_assertions)) { install_logging_hooks(); }
        let list = ["tiny-q5_1", "tiny-q8_0", "base-q5_1", "base-q8_0", "small-q5_1", "small-q8_0"];
        let work_path = std::env::current_dir().unwrap().join("resource");
        let test_path = work_path.join("test.wav");
        let mut res_buf = String::new();
        let threads = get_thread_count(ThreadParm::Max);

        for mode in [ThreadParm::Max, ThreadParm::Half, ThreadParm::Quarter] {
            let threads = get_thread_count(mode);
            for model_name in list {
                let model_path = work_path.join(format!("ggml-{}.bin", model_name));
                for i in 1..4 {
                    let start_time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
                    let params = TranscribeParams { lang: Some("ja"), threads };
                    let result = transcribe(&model_path, &test_path, &params).unwrap_or_else(|e| e);
                    let end_time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
                    let time = (end_time - start_time).as_millis();
                    res_buf.push_str(&format!("{:?},{threads},{i},{model_name},{time},{result}\n", mode));
                }
            }
        }
        println!("\nprams-mode,threads,index,model,time,result\n{res_buf}");
    }

    #[test]
    fn send_test() {
        let t = "Hellow world!";
        if let Ok(mut clipboard) = Clipboard::new() {
            clipboard.set_text(t.to_string()).ok();
        }
    }
}
