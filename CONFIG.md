# Configuration Guide

`voime` uses a TOML file for configuration.

## File Location

On Linux, the configuration file is located at:
`~/.config/voime/config.toml`

The file is automatically created with default values on the first run.

## Options

| Key | Type | Default | Description |
|:---|:---:|:---:|:---|
| `model_path` | string? | `null` | Path to a custom Whisper model (`.bin`). If not set, it uses the default path in `~/.local/share/voime/`. |
| `language` | string | `"ja"` | Transcription language code (e.g., `"ja"`, `"en"`, `"auto"`). |
| `dark_mode` | boolean | `true` | Enables GTK dark theme. |
| `opacity` | float | `0.9` | Sets the opacity level (0.0 to 1.0) of the UI background. |
| `auto_copy` | boolean | `true` | Automatically copies transcribed text to the clipboard upon completion. |
| `auto_start_record` | boolean | `false` | Automatically starts recording when the application opens and the model is ready. |
| `key_record` | list(string) | `["q", "Return"]` | List of key names to start or stop recording. |
| `key_copy` | list(string) | `["c", "y", "space"]` | List of key names for manual clipboard copy. |

## Example `config.toml`

```toml
model_path = "/home/user/models/ggml-base-q8_0.bin"
dark_mode = true
transparent = true
auto_copy = true
key_record = "q"
key_copy = ["c", "y", "space"]
```

## Key Names

Key names follow GTK/GDK naming conventions (e.g., `"q"`, `"space"`, `"Escape"`, `"Return"`).
