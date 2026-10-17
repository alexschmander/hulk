use super::server::{Operation, State, id, string};
use color_eyre::{
    Result,
    eyre::{bail, ensure, eyre},
};
use image::{Rgb, RgbImage};
use serde_json::{Value, json};
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

pub(super) async fn execute(
    state: &Arc<State>,
    tool: &str,
    args: &Value,
    operation: &Operation,
) -> Result<Value> {
    if tool == "video_capture" {
        return video(state, args, operation).await;
    }
    let view = &args["view"];
    let path = state.artifacts.join(format!("{}.png", id()));
    operation.stage("rendering");
    let result = match view["kind"].as_str().unwrap() {
        "scene" => {
            let worker = state.worker(view).await?;
            worker
                .call(
                    "render",
                    json!({"camera":view["camera"],"size":args["size"],"path":path}),
                )
                .await?
        }
        "plot" => {
            let result=super::rosz::read(state,&json!({"target_id":view["target_id"],"sources":view["sources"],"query":{"kind":"history","range":view["range"]},"allow_recording_fallback":view["allow_recording_fallback"].as_bool().unwrap_or(false),"limit":10000,"max_bytes":1048576})).await?;
            state
                .queries
                .lock()
                .unwrap()
                .release(result["query_id"].as_str().unwrap());
            ensure!(
                result["truncated"] == false,
                "Plot exceeds inline sample budget; narrow the interval"
            );
            plot(&path, args, &result, &view["range"])?;
            json!({"query_id":result["query_id"],"warnings":["Historical coverage may contain retention or transport gaps; see rosz_read for coverage details."]})
        }
        "image" => {
            let result=super::rosz::read(state,&json!({"target_id":view["target_id"],"sources":[view["source"]],"query":{"kind":"latest"},"max_bytes":1048576})).await?;
            let sample = result["samples"]
                .as_array()
                .and_then(|samples| samples.first())
                .ok_or_else(|| eyre!("No published image available"))?;
            let overflow;
            let value = if sample["value_status"] == "too_large" {
                let uri = string(&sample["artifact"], "uri")?;
                let path = state
                    .artifact_index
                    .lock()
                    .unwrap()
                    .get(uri)
                    .cloned()
                    .ok_or_else(|| eyre!("Missing image value artifact"))?;
                overflow = serde_json::from_slice::<Value>(&std::fs::read(path)?)?;
                &overflow
            } else {
                &sample["value"]
            };
            let data = value
                .get("data")
                .unwrap_or(value)
                .as_array()
                .ok_or_else(|| eyre!("Image source must contain encoded bytes or a data array"))?;
            let bytes: Vec<u8> = data
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| n.try_into().ok())
                        .ok_or_else(|| eyre!("Invalid image byte"))
                })
                .collect::<Result<_>>()?;
            let image = if let (Some(width), Some(height)) =
                (value["width"].as_u64(), value["height"].as_u64())
            {
                ensure!(
                    width <= 8192 && height <= 8192,
                    "Image dimensions exceed decode budget"
                );
                ensure!(
                    value["encoding"] == "rgb8",
                    "Raw image encoding must be rgb8"
                );
                image::DynamicImage::ImageRgb8(
                    RgbImage::from_raw(width.try_into()?, height.try_into()?, bytes)
                        .ok_or_else(|| eyre!("Invalid image dimensions"))?,
                )
            } else {
                let mut reader =
                    image::ImageReader::new(std::io::Cursor::new(&bytes)).with_guessed_format()?;
                let mut limits = image::Limits::default();
                limits.max_image_width = Some(8192);
                limits.max_image_height = Some(8192);
                limits.max_alloc = Some(256 * 1024 * 1024);
                reader.limits(limits);
                reader.decode()?
            };
            image
                .thumbnail(
                    args["size"]["width"].as_u64().unwrap_or(1280) as u32,
                    args["size"]["height"].as_u64().unwrap_or(720) as u32,
                )
                .save(&path)?;
            json!({"sample":sample})
        }
        "map" => {
            let samples = map(state, view, args, &path).await?;
            json!({"source_samples":samples,"warnings":["Map layers use coordinates as published without frame conversion. Truth overlays use the world frame."]})
        }
        _ => bail!("Unknown view"),
    };
    let mut output = json!({"artifact":state.artifact(&path,"On-demand simulator inspection")?,"snapshot_id":id(),"source_samples":result.get("source_samples").cloned().unwrap_or_else(||json!(result.get("sample").into_iter().cloned().collect::<Vec<_>>())),"warnings":result.get("warnings").cloned().unwrap_or_else(||json!([]))});
    for key in ["query_id", "simulation_time_ns"] {
        if let Some(value) = result.get(key) {
            output[key] = value.clone();
        }
    }
    Ok(output)
}
async fn video(state: &Arc<State>, args: &Value, operation: &Operation) -> Result<Value> {
    let worker = state.worker(args).await?;
    let duration = string(args, "duration_ns")?.parse::<i64>()?;
    ensure!(
        (1..=30_000_000_000).contains(&duration),
        "Clips are limited to 30 simulated seconds"
    );
    let fps = args["fps"].as_u64().unwrap_or(15);
    let period = 1_000_000_000 / fps as i64;
    let start = worker.time();
    let end = start
        .checked_add(duration)
        .ok_or_else(|| eyre!("Clip interval overflow"))?;
    let path = state.artifacts.join(format!("{}.gif", id()));
    let frame_path = state.artifacts.join(format!("{}.png", id()));
    let mut encoder = image::codecs::gif::GifEncoder::new(std::fs::File::create(&path)?);
    encoder.set_repeat(image::codecs::gif::Repeat::Infinite)?;
    let deadline =
        Instant::now() + Duration::from_millis(args["wall_timeout_ms"].as_u64().unwrap());
    let mut next = start;
    let mut frames = 0;
    let mut pending: Option<(image::RgbaImage, i64)> = None;
    let mut skipped = 0;
    operation.stage("armed");
    loop {
        if operation.cancel.is_cancelled()
            || Instant::now() >= deadline
            || worker.stopped.load(std::sync::atomic::Ordering::Acquire)
        {
            break;
        }
        let now = worker.time();
        if now >= end {
            break;
        }
        if now >= next {
            let result = worker
                .call(
                    "render",
                    json!({"camera":args["camera"],"size":args["size"],"path":frame_path}),
                )
                .await?;
            let actual = string(&result, "simulation_time_ns")?.parse::<i64>()?;
            skipped += (actual - next).max(0) / period;
            next = actual + period;
            let frame = image::open(&frame_path)?.to_rgba8();
            if let Some((previous, timestamp)) = pending.replace((frame, actual)) {
                encoder.encode_frame(image::Frame::from_parts(
                    previous,
                    0,
                    0,
                    image::Delay::from_numer_denom_ms(
                        ((actual - timestamp).max(1) / 1_000_000).max(1) as u32,
                        1,
                    ),
                ))?;
            }
            frames += 1;
        } else {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
    if let Some((previous, timestamp)) = pending {
        let delay = (worker.time().min(end) - timestamp).max(1) / 1_000_000;
        encoder.encode_frame(image::Frame::from_parts(
            previous,
            0,
            0,
            image::Delay::from_numer_denom_ms(delay.max(1) as u32, 1),
        ))?;
    }
    drop(encoder);
    let _ = std::fs::remove_file(&frame_path);
    ensure!(frames > 0, "No frames captured");
    Ok(
        json!({"artifact":state.artifact(&path,"Simulator clip, animated GIF")?,"frame_count":frames,"start_simulation_ns":start.to_string(),"end_simulation_ns":worker.time().min(end).to_string(),"partial":worker.time()<end,"warnings":if skipped>0{vec![format!("Rendering missed {skipped} requested frame times")]}else{vec![]}}),
    )
}
fn canvas(args: &Value) -> RgbImage {
    RgbImage::from_pixel(
        args["size"]["width"].as_u64().unwrap_or(1280) as u32,
        args["size"]["height"].as_u64().unwrap_or(720) as u32,
        Rgb([245, 247, 249]),
    )
}
fn line(image: &mut RgbImage, a: (i32, i32), b: (i32, i32), color: [u8; 3]) {
    // Bound raster work even for wildly out-of-range published coordinates.
    let clip = |(x, y): (i32, i32)| {
        (
            x.clamp(0, image.width() as i32 - 1),
            y.clamp(0, image.height() as i32 - 1),
        )
    };
    let a = clip(a);
    let b = clip(b);
    let (mut x, mut y) = a;
    let dx = (b.0 - x).abs();
    let dy = -(b.1 - y).abs();
    let sx = if x < b.0 { 1 } else { -1 };
    let sy = if y < b.1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        if x >= 0 && y >= 0 && (x as u32) < image.width() && (y as u32) < image.height() {
            image.put_pixel(x as u32, y as u32, Rgb(color));
        }
        if (x, y) == b {
            break;
        }
        let e = 2 * error;
        if e >= dy {
            error += dy;
            x += sx;
        }
        if e <= dx {
            error += dx;
            y += sy;
        }
    }
}
fn text(image: &mut RgbImage, x: i32, y: i32, value: &str) {
    use font8x8::UnicodeFonts;
    for (index, c) in value.chars().take(256).enumerate() {
        if let Some(glyph) = font8x8::BASIC_FONTS.get(c) {
            for (dy, bits) in glyph.iter().enumerate() {
                for dx in 0..8 {
                    if bits & (1 << dx) != 0 {
                        let px = x + index as i32 * 8 + dx;
                        let py = y + dy as i32;
                        if px >= 0
                            && py >= 0
                            && (px as u32) < image.width()
                            && (py as u32) < image.height()
                        {
                            image.put_pixel(px as u32, py as u32, Rgb([30, 38, 50]));
                        }
                    }
                }
            }
        }
    }
}
fn number(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        .filter(|n| n.is_finite())
}
fn label(v: &Value) -> String {
    v["variant_name"]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| v.to_string())
}
fn plot(path: &Path, args: &Value, result: &Value, range: &Value) -> Result<()> {
    let mut image = canvas(args);
    let w = image.width() as i32;
    let h = image.height() as i32;
    let start = string(range, "start_ns")?.parse::<i64>()?;
    let end = string(range, "end_ns")?.parse::<i64>()?;
    let sources = args["view"]["sources"].as_array().unwrap();
    let row = (h - 55) / sources.len() as i32;
    text(
        &mut image,
        16,
        12,
        &format!("{}  [{} ns, {} ns)", string(range, "clock")?, start, end),
    );
    for (index, source) in sources.iter().enumerate() {
        let top = 40 + index as i32 * row;
        let bottom = top + row - 20;
        let samples: Vec<_> = result["samples"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|s| {
                s["topic"] == source["topic"]
                    && s["path"].as_str().unwrap_or("") == source["path"].as_str().unwrap_or("")
            })
            .collect();
        text(
            &mut image,
            16,
            top,
            &format!(
                "{} {}",
                source["topic"].as_str().unwrap(),
                source["path"].as_str().unwrap_or("")
            ),
        );
        line(&mut image, (16, bottom), (w - 16, bottom), [150, 160, 170]);
        let numeric: Vec<_> = samples.iter().filter_map(|s| number(&s["value"])).collect();
        let min = numeric.iter().copied().fold(f64::INFINITY, f64::min);
        let max = numeric.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut previous = None;
        for sample in samples {
            let timestamp = string(sample, "selected_time_ns")?.parse::<i64>()?;
            let x =
                16 + ((timestamp - start) as f64 / (end - start) as f64 * (w - 32) as f64) as i32;
            if let Some(value) = number(&sample["value"]) {
                let y = bottom
                    - 5
                    - ((value - min) / (max - min).max(1e-12) * (row - 45) as f64) as i32;
                if let Some(point) = previous {
                    line(&mut image, point, (x, y), [26, 100, 190]);
                }
                previous = Some((x, y));
            } else {
                let name = label(&sample["value"]);
                let color = name
                    .bytes()
                    .fold(29u32, |h, b| h.wrapping_mul(31) + b as u32);
                line(
                    &mut image,
                    (x, top + 18),
                    (x, bottom - 2),
                    [(color >> 16) as u8, color as u8, (color >> 8) as u8],
                );
                if previous.is_none_or(|(px, _)| x - px > 100) {
                    text(&mut image, x, top + 20, &name);
                    previous = Some((x, 0));
                }
            }
        }
        if !numeric.is_empty() {
            text(
                &mut image,
                16,
                bottom - 12,
                &format!("min {min:.4}  max {max:.4}"),
            );
        }
    }
    image.save(path)?;
    Ok(())
}
async fn map(state: &Arc<State>, view: &Value, args: &Value, path: &Path) -> Result<Vec<Value>> {
    let mut source_samples = Vec::new();
    let target = state.target(view).await?;
    let mut image = canvas(args);
    let width = image.width() as i32;
    let height = image.height() as i32;
    let scale = width.min(height) as f64 / 12.0;
    text(&mut image, 12, 12, "Robot framework map, metres");
    line(
        &mut image,
        (width / 2, 35),
        (width / 2, height - 20),
        [180, 190, 200],
    );
    line(
        &mut image,
        (20, height / 2),
        (width - 20, height / 2),
        [180, 190, 200],
    );
    for layer in view["layers"].as_array().unwrap() {
        let topic = layer
            .as_str()
            .ok_or_else(|| eyre!("Map layer must be a topic"))?;
        let result = super::rosz::read(
            state,
            &json!({"target_id":target.id,"sources":[{"topic":topic}],"query":{"kind":"latest"}}),
        )
        .await?;
        let value = result["samples"]
            .as_array()
            .and_then(|s| s.first())
            .and_then(|s| s.get("value"))
            .ok_or_else(|| eyre!("No data for map layer {topic}"))?;
        source_samples.extend(result["samples"].as_array().unwrap().iter().cloned());
        // Map layers explicitly select two-dimensional point records; no inferred transform.
        let point = value.get("position").unwrap_or(value);
        let x = number(&point["x"]).ok_or_else(|| {
            eyre!(
                "Map layer {topic} must expose x/y or position.x/position.y in the displayed frame"
            )
        })?;
        let y =
            number(&point["y"]).ok_or_else(|| eyre!("Map layer {topic} has no y coordinate"))?;
        let px = (width as f64 / 2.0 + x * scale).clamp(-8192.0, 8192.0) as i32;
        let py = (height as f64 / 2.0 - y * scale).clamp(-8192.0, 8192.0) as i32;
        line(&mut image, (px - 5, py), (px + 5, py), [30, 110, 200]);
        line(&mut image, (px, py - 5), (px, py + 5), [30, 110, 200]);
        text(&mut image, px + 8, py, topic);
    }
    if view["truth_overlays"] == true {
        let scene = target.worker.call("scene", json!({})).await?;
        for entity in scene["objects"].as_array().unwrap() {
            let p = &entity["pose"]["position_m"];
            let px =
                (width as f64 / 2.0 + p[0].as_f64().unwrap() * scale).clamp(-8192.0, 8192.0) as i32;
            let py = (height as f64 / 2.0 - p[1].as_f64().unwrap() * scale).clamp(-8192.0, 8192.0)
                as i32;
            line(
                &mut image,
                (px - 4, py - 4),
                (px + 4, py + 4),
                [200, 50, 50],
            );
            text(&mut image, px + 6, py + 6, "truth");
        }
    }
    image.save(path)?;
    Ok(source_samples)
}
