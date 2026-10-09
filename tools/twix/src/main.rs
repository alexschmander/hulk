use std::{env::current_dir, path::PathBuf, str::FromStr, sync::Arc};

use clap::Parser;
use color_eyre::{
    Result,
    eyre::{Context as _, ContextCompat as _},
};
use configuration::{
    Configuration,
    keybind_plugin::{self, KeybindSystem},
    keys::KeybindAction,
};
use eframe::{
    App, CreationContext, Frame, NativeOptions, Renderer, Storage,
    egui::{CentralPanel, Layout, Panel as EguiPanel, Ui},
    egui_wgpu::{WgpuConfiguration, WgpuSetup},
    emath::Align,
    run_native,
};
use layout::{FocusDirection, TwixLayout};
use log::{error, warn};
use panels::{
    AudioPanel, BehaviorTreePanel, ImagePanel, Map3DPanel, MapPanel, ParameterPanel, PlotPanel,
    SimulatorPanel, TextPanel, TimelinePanel,
};
use repository::{Repository, inspect_version::check_for_update};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
use visuals::Visuals;

use crate::backend::RobotBackend;

mod backend;
mod configuration;
mod graph;
mod layout;
mod namespace_discovery;
mod panel;
mod panels;
mod presets;
mod repaint;
mod replay;
mod selectable_panel_macro;
mod topic_source;
mod visuals;

impl_selectable_panel!(
    TextPanel,
    ImagePanel,
    MapPanel,
    Map3DPanel,
    ParameterPanel,
    AudioPanel,
    PlotPanel,
    BehaviorTreePanel,
    TimelinePanel,
    SimulatorPanel
);

#[derive(Debug, Clone, clap::Parser)]
struct Arguments {
    /// Target ROS-Z namespace, for example /42.
    namespace: Option<String>,

    /// Router endpoint passed to ROS-Z, for example tcp/127.0.0.1:7447.
    #[arg(long)]
    router: Option<String>,

    /// Global Zenoh key prefix, including discovery and raw SDK traffic.
    #[arg(long)]
    zenoh_namespace: Option<String>,

    /// Alternative repository root for local Twix version checks.
    #[arg(long)]
    repository_root: Option<PathBuf>,

    /// Start with one blank workspace, ignoring the saved session.
    #[arg(long)]
    clear: bool,
}

struct TwixApp {
    layout: TwixLayout,
    namespace_editor: String,
    visual: Visuals,
    backend: Arc<RobotBackend>,
    runtime: tokio::runtime::Runtime,
    #[cfg(feature = "simulator")]
    simulator_router: Option<Arc<RobotBackend>>,
    #[cfg(feature = "simulator")]
    pending_backend: Option<(String, tokio::task::JoinHandle<Result<RobotBackend>>)>,
}

impl TwixApp {
    fn create(
        creation_context: &CreationContext,
        arguments: Arguments,
        runtime: tokio::runtime::Runtime,
        backend: Arc<RobotBackend>,
        configuration: Configuration,
    ) -> Self {
        if let Some(renderer) = &creation_context.wgpu_render_state {
            creation_context.egui_ctx.data_mut(|data| {
                data.insert_temp(
                    eframe::egui::Id::new(panels::simulator::RENDER_STATE_ID),
                    renderer.clone(),
                )
            });
        }
        let namespace_editor = backend.namespace();
        if let Some(render_state) = &creation_context.wgpu_render_state {
            creation_context.egui_ctx.data_mut(|data| {
                data.insert_temp(eframe::egui::Id::new("render_state"), render_state.clone());
            });
        }
        if let Some(scope) = backend.transport_scope() {
            log::info!("Zenoh transport namespace: {scope}");
        }

        let layout = TwixLayout::load(
            creation_context.storage,
            &creation_context.egui_ctx,
            &backend,
            arguments.clear,
        );

        keybind_plugin::register(&creation_context.egui_ctx);
        creation_context
            .egui_ctx
            .set_keybinds(Arc::new(configuration.keys));

        let visual = creation_context
            .storage
            .and_then(|storage| storage.get_string("style"))
            .and_then(|theme| Visuals::from_str(&theme).ok())
            .unwrap_or(Visuals::Dark);
        visual.set_visual(&creation_context.egui_ctx);

        Self {
            #[cfg(feature = "simulator")]
            simulator_router: (backend.transport_scope() == Some(simulate::ZENOH_NAMESPACE))
                .then(|| backend.clone()),
            #[cfg(feature = "simulator")]
            pending_backend: None,
            layout,
            namespace_editor,
            visual,
            backend,
            runtime,
        }
    }
    fn namespace_completions(&self) -> Vec<String> {
        #[cfg(feature = "simulator")]
        if let Some(root) = &self.simulator_router {
            return root.namespace_completions();
        }
        self.backend.namespace_completions()
    }

    #[cfg(feature = "simulator")]
    fn synchronize_simulator_target(&mut self, context: &eframe::egui::Context) {
        let Some(root) = &self.simulator_router else {
            return;
        };
        let namespace = self.backend.namespace();
        let number = simulate::robot_id(&namespace);
        self.layout.select_simulator_robot(number);
        let scope = number.map_or_else(
            || simulate::ZENOH_NAMESPACE.to_owned(),
            simulate::transport_scope,
        );
        if self
            .pending_backend
            .as_ref()
            .is_some_and(|(pending, _)| *pending != scope)
            && let Some((_, task)) = self.pending_backend.take()
        {
            task.abort();
        }
        if self
            .pending_backend
            .as_ref()
            .is_some_and(|(_, task)| task.is_finished())
        {
            let (_, task) = self.pending_backend.take().unwrap();
            match self.runtime.block_on(task) {
                Ok(Ok(backend)) => {
                    self.backend = Arc::new(backend);
                    self.layout.reconnect(&self.backend, context);
                }
                Ok(Err(error)) => log::error!("Switching simulator connection: {error:#}"),
                Err(error) => log::error!("Switching simulator task: {error}"),
            }
        }
        if self.backend.transport_scope() != Some(scope.as_str()) && self.pending_backend.is_none()
        {
            let runtime = self.runtime.handle().clone();
            let router = root.router().map(str::to_owned);
            let requested_scope = scope.clone();
            let repaint = context.clone();
            self.pending_backend = Some((
                scope,
                self.runtime.spawn(async move {
                    let result =
                        RobotBackend::new_scoped(runtime, router, namespace, Some(requested_scope))
                            .await;
                    repaint.request_repaint();
                    result
                }),
            ));
        }
    }
}

impl App for TwixApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut Frame) {
        let runtime_handle = self.runtime.handle().clone();
        let _runtime_guard = runtime_handle.enter();
        let context = ui.ctx().clone();
        let shortcuts_enabled = !self.layout.dialog_open();

        EguiPanel::top("top_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    if !ui.memory(|memory| memory.focused().is_some()) {
                        self.namespace_editor = self.backend.namespace();
                    }
                    ui.label("Namespace:");
                    let namespaces = self.namespace_completions();
                    let namespace_response = ui.add(
                        hulk_widgets::CompletionEdit::new(
                            ui.id().with("namespace"),
                            &namespaces,
                            &mut self.namespace_editor,
                        )
                        .request_focus(
                            shortcuts_enabled
                                && context.keybind_pressed(KeybindAction::FocusNamespace),
                        ),
                    );
                    if namespace_response.has_focus() {
                        context.request_repaint_after(std::time::Duration::from_millis(250));
                    }
                    if (namespace_response.changed() || namespace_response.lost_focus())
                        && self.namespace_editor != self.backend.namespace()
                        && let Err(error) =
                            self.backend.set_namespace(self.namespace_editor.clone())
                    {
                        log::error!("failed to set namespace: {error:#}");
                        self.namespace_editor = self.backend.namespace();
                    }
                });

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.menu_button("Settings", |ui| {
                        ui.menu_button("Theme", |ui| {
                            ui.vertical(|ui| {
                                for visual in Visuals::iter() {
                                    if ui.button(visual.to_string()).clicked() {
                                        self.visual = visual;
                                        self.visual.set_visual(&context);
                                    }
                                }
                            })
                        });
                    });
                });
            })
        });

        self.backend.replay().update(&context, &self.backend);

        #[cfg(feature = "simulator")]
        self.synchronize_simulator_target(&context);

        CentralPanel::default().show(ui, |ui| {
            let layout = &mut self.layout;
            if shortcuts_enabled {
                if context.keybind_pressed(KeybindAction::FocusPanel) {
                    layout.open_selector(&context);
                }
                for (action, direction) in [
                    (KeybindAction::FocusLeft, FocusDirection::Left),
                    (KeybindAction::FocusBelow, FocusDirection::Below),
                    (KeybindAction::FocusAbove, FocusDirection::Above),
                    (KeybindAction::FocusRight, FocusDirection::Right),
                ] {
                    if context.keybind_pressed(action) {
                        layout.focus(direction, &context);
                    }
                }
                if context.keybind_pressed(KeybindAction::FocusTopic) {
                    layout.focus_topic(&context);
                }
                if context.keybind_pressed(KeybindAction::OpenSplit) {
                    layout.open_split(&self.backend, &context);
                }
                if context.keybind_pressed(KeybindAction::OpenTab) {
                    layout.open_tab(&self.backend, &context);
                }
                if context.keybind_pressed(KeybindAction::DuplicateTab) {
                    layout.duplicate_focused(&self.backend, &context);
                }
                if context.keybind_pressed(KeybindAction::CloseTab) {
                    layout.close_focused(&self.backend, &context);
                }
                if context.keybind_pressed(KeybindAction::CloseAll) {
                    layout.reset(&self.backend, &context);
                }
            }
            layout.ui(ui, &self.backend);
        });
        self.layout.dialogs(&context);
        self.backend.replay().dispatch(&context, &self.backend);
    }

    fn save(&mut self, storage: &mut dyn Storage) {
        self.layout.save(storage);
        storage.set_string("namespace", self.backend.namespace());
        storage.set_string("style", self.visual.to_string());
    }
}

fn setup_logger() -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,bevy_render=warn"));
    let layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_level(true)
        .with_file(false)
        .with_line_number(true)
        .compact();

    tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .try_init()
        .wrap_err("failed to initialize tracing subscriber")?;

    Ok(())
}

fn main() -> eframe::Result<()> {
    color_eyre::install().expect("failed to install color-eyre");
    setup_logger().expect("failed to setup logger");
    let arguments = Arguments::parse();
    let repository = arguments
        .repository_root
        .clone()
        .map(Repository::new)
        .map(Ok)
        .unwrap_or_else(|| {
            let current_directory = current_dir().wrap_err("failed to get current directory")?;
            Repository::find_root(current_directory).wrap_err("failed to find repository root")
        });
    match &repository {
        Ok(repository) => {
            if let Err(error) = check_for_update(
                env!("CARGO_PKG_VERSION"),
                repository.root.join("tools/twix/Cargo.toml"),
                "twix",
            ) {
                error!("{error:#}");
            }
        }
        Err(error) => {
            warn!("{error:#}");
        }
    }

    let configuration = Configuration::load()
        .unwrap_or_else(|error| panic!("failed to load configuration: {error}"));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build Tokio runtime");
    let runtime_handle = runtime.handle().clone();

    let mut wgpu_options = WgpuConfiguration::default();
    if let WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        let previous = setup.device_descriptor.clone();
        setup.device_descriptor = Arc::new(move |adapter| {
            let mut descriptor = previous(adapter);
            descriptor
                .required_limits
                .max_storage_buffers_per_shader_stage = 9;
            descriptor
        });
    }

    run_native(
        "Twix",
        {
            #[allow(unused_mut)]
            let mut options = NativeOptions {
                renderer: Renderer::Wgpu,
                wgpu_options,
                ..Default::default()
            };
            #[cfg(feature = "simulator")]
            simulate::configure_renderer(&mut options);
            options
        },
        Box::new(move |creation_context| {
            egui_extras::install_image_loaders(&creation_context.egui_ctx);
            egui_material_icons::initialize(&creation_context.egui_ctx);
            let namespace = arguments
                .namespace
                .clone()
                .or_else(|| creation_context.storage?.get_string("namespace"))
                .unwrap_or_else(|| "/".to_string());
            let backend = runtime.block_on(RobotBackend::new_scoped(
                runtime_handle.clone(),
                arguments.router.clone(),
                namespace,
                arguments.zenoh_namespace.clone(),
            ))?;
            Ok(Box::new(TwixApp::create(
                creation_context,
                arguments.clone(),
                runtime,
                Arc::new(backend),
                configuration,
            )))
        }),
    )
}
