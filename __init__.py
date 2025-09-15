# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import bpy
import time
import os.path
import math
from collections import deque

# Optional deps for LIVE mode
try:
    import numpy as np
    import sounddevice as sd
    _SAV_HAVE_SD = True
except Exception:
    _SAV_HAVE_SD = False

# -----------------------------
# Live audio globals/state
# -----------------------------
_sav_live_stream = None
_sav_live_should_run = False
_sav_live_samplerate = 48000
_sav_live_last_chunk = None  # numpy array (frames, channels)
_sav_live_targets = []       # [{'name': str, 'path': str, 'axis': int}]
_sav_live_prev_vals = None   # np array len(targets) for smoothing
_sav_live_freq_edges = None  # list of (lo, hi) per target
_sav_audio_ring = deque()    # deque of 1D float32 arrays (mono samples)
_sav_audio_ring_len = 0      # total samples in ring
_SAV_MAX_RING_SAMPLES = 131072
_sav_silence_last_loud_time = 0.0  # timestamp of last above-threshold audio


def bake_sound_to_channels(obj, context, path, lo, hi, intensity_multiplier,
                           anim_loc, anim_rot, anim_scale, anim_x, anim_y,
                           anim_z, additive_mode=False):
    """
    Bakes sound data to specified transform channels of a given object.
    This function handles creating animation data, switching to the graph editor context,
    baking the sound to f-curves, and applying an intensity multiplier.
    """
    # Ensure animation data exists
    if not obj.animation_data:
        obj.animation_data_create()
    if not obj.animation_data.action:
        obj.animation_data.action = bpy.data.actions.new(name=f"{obj.name}Action")

    action = obj.animation_data.action
    fcurves = action.fcurves

    # Determine which channels to animate
    props_to_animate = []
    if anim_loc: props_to_animate.append('location')
    if anim_rot: props_to_animate.append('rotation_euler')
    if anim_scale: props_to_animate.append('scale')

    axes_to_animate = []
    if anim_x: axes_to_animate.append(0)
    if anim_y: axes_to_animate.append(1)
    if anim_z: axes_to_animate.append(2)

    if not props_to_animate or not axes_to_animate:
        return  # Nothing to do

    # If not in additive mode, remove any existing f-curves for the channels
    # we are about to bake to ensure a clean bake.
    if not additive_mode:
        for fc in list(fcurves):
            if fc.data_path in props_to_animate and fc.array_index in axes_to_animate:
                fcurves.remove(fc)

    # Find or create the f-curves to be animated
    fcs_to_bake = []
    for prop in props_to_animate:
        for axis in axes_to_animate:
            fc = fcurves.find(prop, index=axis)
            if not fc:
                obj.keyframe_insert(data_path=prop, index=axis, frame=context.scene.frame_current, group=prop)
                fc = fcurves.find(prop, index=axis)
            if fc:
                fcs_to_bake.append(fc)

    if not fcs_to_bake:
        print(f"Warning: Could not create any F-Curves for {obj.name}")
        return

    # Switch area to Graph Editor for the sound-to-fcurve operator
    original_area_type = context.area.type
    context.area.type = 'GRAPH_EDITOR'

    try:
        # Deselect all f-curves in the action
        for fc in fcurves:
            fc.select = False

        # Select only the curves we want to bake
        for fc in fcs_to_bake:
            fc.select = True

        # Bake sound to all selected F-Curves
        bpy.ops.graph.sound_to_samples(filepath=path, low=lo, high=hi, use_additive=additive_mode)
        bpy.ops.graph.samples_to_keys()

        # Apply intensity multiplier and deselect
        for fc in fcs_to_bake:
            if intensity_multiplier != 1.0:
                for kf in fc.keyframe_points:
                    kf.co.y *= intensity_multiplier
                    kf.handle_left.y *= intensity_multiplier
                    kf.handle_right.y *= intensity_multiplier
            fc.select = False
    finally:
        # Restore the original area type to not disrupt the user's layout
        context.area.type = original_area_type

class SimpleAudioVisualizerProperties(bpy.types.PropertyGroup):
    # --- Source selection ---
    input_mode: bpy.props.EnumProperty(
        name="Input Source",
        description="Choose where the audio comes from",
        items=[
            ('FILE', 'Audio File', 'Bake from an audio file'),
            ('MIC', 'Microphone (Input)', 'Use a live microphone/input device'),
            ('SPEAKERS', 'Speakers (Loopback)', 'Use live system audio (Windows WASAPI loopback)')
        ],
        default='FILE'
    )
    # Device selectors (populated at draw time)
    def _sav_input_devices(self, context):
        items = []
        if not _SAV_HAVE_SD:
            return items
        try:
            devs = sd.query_devices()
            has = sd.query_hostapis()
            for i, d in enumerate(devs):
                if int(d.get('max_input_channels', 0)) > 0:
                    ha = has[d['hostapi']]['name']
                    label = f"{i}:{d['name']} ({ha})"
                    items.append((str(i), label, ""))
        except Exception:
            pass
        return items
    def _sav_wasapi_speakers(self, context):
        items = []
        if not _SAV_HAVE_SD:
            return items
        try:
            devs = sd.query_devices()
            has = sd.query_hostapis()
            for i, d in enumerate(devs):
                if int(d.get('max_output_channels', 0)) > 0:
                    ha = has[d['hostapi']]['name']
                    if 'WASAPI' in ha.upper():
                        label = f"{i}:{d['name']} ({ha})"
                        items.append((str(i), label, ""))
        except Exception:
            pass
        return items
    mic_device: bpy.props.EnumProperty(
        name="Input Device",
        description="Microphone / line-in device to capture",
        items=_sav_input_devices
    )
    spk_device: bpy.props.EnumProperty(
        name="Speakers (WASAPI)",
        description="WASAPI output device to loopback capture",
        items=_sav_wasapi_speakers
    )
    live_update_interval: bpy.props.FloatProperty(
        name="Live Update (s)", default=0.03, min=0.01, max=1.0,
        description="How often to update the visualization from live audio"
    )
    fft_size: bpy.props.IntProperty(
        name="FFT Size", default=2048, min=256, max=16384,
        description="FFT window size for live spectrum"
    )
    smoothing: bpy.props.FloatProperty(
        name="Smoothing", default=0.50, min=0.0, max=0.98,
        description="Smoothing factor for live values (0=no smoothing, 0.9=heavy smoothing)"
    )
    reset_on_silence: bpy.props.BoolProperty(
        name="Reset on silence",
        default=True,
        description="When audio level stays below the threshold for a short time, reset properties to their baseline to avoid jitter"
    )
    silence_threshold: bpy.props.FloatProperty(
        name="Silence threshold (RMS)",
        default=0.01, min=0.0, max=0.2,
        description="Signal RMS level below which audio is considered silent"
    )
    silence_hold: bpy.props.FloatProperty(
        name="Hold (s)",
        default=0.35, min=0.0, max=5.0,
        description="How long audio must remain below the threshold before resetting"
    )
    # Live routing: drive selected objects or a GN bands node group (via empties)
    live_route: bpy.props.EnumProperty(
        name="Live Routing",
        description="Choose where live audio is sent",
        items=[
            ('OBJECTS', 'Selected Objects', 'Drive selected objects\' transforms'),
            ('GN_BANDS', 'Geometry Nodes Bands', 'Drive a GN node group with vector outputs using controller empties')
        ],
        default='OBJECTS'
    )
    gn_bands: bpy.props.IntProperty(
        name="Bands",
        description="Number of band outputs in the GN group",
        default=8, min=1, max=64
    )
    gn_output_mode: bpy.props.EnumProperty(
        name="Output Mode",
        description="How to expose bands to Geometry Nodes",
        items=[
            ('SOCKETS', 'Sockets per Band', 'One output per band (Vector/Float/Int)'),
            ('POINTS', 'Points Geometry', 'Single Geometry with one point per band; use Index and Position in GN')
        ],
        default='SOCKETS'
    )
    gn_output_type: bpy.props.EnumProperty(
        name="Output Type",
        description="Type of outputs created in the bands node group",
        items=[
            ('VECTOR', 'Vector', 'Vector outputs (XYZ)'),
            ('FLOAT', 'Float', 'Single scalar output per band'),
            ('INT', 'Integer', 'Single integer output per band')
        ],
        default='VECTOR'
    )
    gn_collection_hide: bpy.props.BoolProperty(
        name="Hide Controller Empties",
        default=True,
        description="Hide the controller empties collection in viewport and render"
    )
    # Live status (shown in UI and set by operators)
    live_active: bpy.props.BoolProperty(name="Live Active", default=False, options={'HIDDEN'})
    live_device_label: bpy.props.StringProperty(name="Live Device", default="", options={'HIDDEN'})
    live_status: bpy.props.StringProperty(name="Live Status", default="", options={'HIDDEN'})
    columns: bpy.props.IntProperty(
        name="Number of cubes",
        default=20,
        min=1,
        description="Number of created objects. Higher values will increase calculation time."
    )
    path: bpy.props.StringProperty(
        name="Audio File",
        subtype='FILE_PATH',
        description="Set the path to the audio file."
    )
    mirror_mode: bpy.props.BoolProperty(
        name="Mirror Mode",
        default=False,
        description="Offsets the origins of the cubes to create a mirror effect."
    )
    intensity: bpy.props.FloatProperty(
        name="Audio Visualizer Multiplier",
        default=1.0,
        min=0.0,
        soft_max=20.0,
        description="Multiplier for the animation's strength"
    )
    animate_location: bpy.props.BoolProperty(
        name="Location",
        default=False,
        description="Animate Location"
    )
    animate_rotation: bpy.props.BoolProperty(
        name="Rotation",
        default=False,
        description="Animate Rotation"
    )
    animate_scale: bpy.props.BoolProperty(
        name="Scale",
        default=True,
        description="Animate Scale"
    )
    animate_axis_x: bpy.props.BoolProperty(
        name="X",
        default=False,
        description="Animate X Axis"
    )
    animate_axis_y: bpy.props.BoolProperty(
        name="Y",
        default=False,
        description="Animate Y Axis"
    )
    animate_axis_z: bpy.props.BoolProperty(
        name="Z",
        default=True,
        description="Animate Z Axis"
    )
    additive_mode: bpy.props.BoolProperty(
        name="Additive",
        default=False,
        description="Add animation from the previous frame to the next, instead of replacing it. Recommended for use with the Rotation channel."
    )
    # Dummy properties for UI tooltips
    info_generate: bpy.props.BoolProperty(
        name="Generate audio visualizer objects in your scene that will be animated based on the selected Audio File. Uses a logarithmic frequency distribution.",
        description="",
        default=False
    )
    info_animate: bpy.props.BoolProperty(
        name="Apply animation data to selected objects in your scene by specifying which transform channels and axes will receive the data. The frequency bands are distributed between the number of selected objects and sorted in alphabetical order.",
        description="",
        default=False
    )



class SimpleAudioVisualizerOperator(bpy.types.Operator):
    bl_idname = "object.simple_audio_visualizer"
    bl_label = "Create Objects"
    bl_description = "Creates new cube objects and animates them to the audio"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sav_props = context.scene.sav_props

        if sav_props.input_mode == 'FILE':
            if not sav_props.path:
                self.report({'ERROR'}, "Please specify an audio file path first.")
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "'Create Objects' is for file baking only. Use 'Start Live' for live audio.")
            return {'CANCELLED'}

        path = os.path.abspath(bpy.path.abspath(sav_props.path))
        if not os.path.exists(path):
            self.report({'ERROR'}, f"Audio file not found: {path}")
            return {'CANCELLED'}

        start_time = time.time()

        columns = sav_props.columns
        intensity = sav_props.intensity
        total_objects = columns
        spectrum_Start = 10    # Frequency in Hz
        spectrum_End = 21000  # Frequency in Hz

        mirror_mode = sav_props.mirror_mode

        maxheight = 6.0  # Default height value

        # For iterative frequency ranges
        l = 1
        h = spectrum_Start
        base = (spectrum_End / spectrum_Start) ** (1 / columns)

        print(f"STARTED (Cubes_Spectrum={columns})")

        bpy.ops.screen.frame_jump(end=False)

        # ---------------------------------------------------------------------
        # Create Visual Elements
        # ---------------------------------------------------------------------
        def create_and_bake_cube(x, y, zmax, lo, hi, intensity_multiplier):
            bpy.ops.mesh.primitive_cube_add(location=(x, y, 0))
            obj = bpy.context.active_object

            if mirror_mode:
                # In mirror mode, the object is centered on the ground plane (Z=0)
                # and scales symmetrically. The default cube creation already
                # places the origin at the center, so no change is needed.
                pass
            else:
                # Default origin at base of the cube
                bpy.context.scene.cursor.location = obj.location
                bpy.context.scene.cursor.location.z -= 1
                bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
                # Move object up to place its base at z=0
                obj.location.z += 1

            # Scale and apply transform
            obj.scale.x = 0.4
            obj.scale.y = 0.4
            obj.scale.z = zmax
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

            bake_sound_to_channels(obj, context, path, lo, hi, intensity_multiplier,
                                   anim_loc=False, anim_rot=False, anim_scale=True,
                                   anim_x=False, anim_y=False, anim_z=True)

        # ---------------------------------------------------------------------
        # Generate the Cubes Along the Spectrum
        # ---------------------------------------------------------------------
        for i in range(columns):
            l = h
            h = round(spectrum_Start * (base ** (i + 1)), 2)
            print(f"     c: {i}   l: {l} Hz    h: {h} Hz")

            height = maxheight
            create_and_bake_cube(0, i, height, l, h, intensity)

            # Optional progress display
            done_percent = (i + 1) / columns * 100
            elapsed_minutes = (time.time() - start_time) / 60
            print(f"{done_percent:.1f}%   done in {elapsed_minutes:.2f} minutes")

        print(f"CREATED {total_objects} OBJECTS")

        # Reset 3D cursor to the world origin
        bpy.context.scene.cursor.location = (0, 0, 0)

        # Deselect the final object to leave a clean state. This is done
        # without operators to be context-agnostic.
        if bpy.context.active_object:
            bpy.context.active_object.select_set(False)
            bpy.context.view_layer.objects.active = None
        return {'FINISHED'}


class VisualizeSelectedObjectsOperator(bpy.types.Operator):
    bl_idname = "object.visualize_selected_objects"
    bl_label = "Visualize Selected Objects"
    bl_description = "Animate the selected objects based on the audio."
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.selected_objects

    def execute(self, context):
        sav_props = context.scene.sav_props
        selected_objects = context.selected_objects

        if sav_props.input_mode == 'FILE':
            if not sav_props.path:
                self.report({'ERROR'}, "Please specify an audio file path first.")
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "'Visualize Selected Objects' is for file baking only. Use 'Start Live' for live audio.")
            return {'CANCELLED'}

        path = os.path.abspath(bpy.path.abspath(sav_props.path))
        if not os.path.exists(path):
            self.report({'ERROR'}, f"Audio file not found: {path}")
            return {'CANCELLED'}

        has_channel = sav_props.animate_location or sav_props.animate_rotation or sav_props.animate_scale
        has_axis = sav_props.animate_axis_x or sav_props.animate_axis_y or sav_props.animate_axis_z

        if not has_channel or not has_axis:
            self.report({'ERROR'}, "Please select at least one transform (Loc/Rot/Scale) and one axis (X/Y/Z).")
            return {'CANCELLED'}

        start_time = time.time()

        columns = len(selected_objects)
        intensity = sav_props.intensity
        spectrum_Start = 10    # Frequency in Hz
        spectrum_End = 21000  # Frequency in Hz

        # For iterative frequency ranges
        l = 1
        h = spectrum_Start
        base = (spectrum_End / spectrum_Start) ** (1 / columns)

        print(f"STARTED (Visualizing {columns} selected objects)")

        # Sort objects by name for consistent frequency distribution
        # Jump to the first frame to ensure sound baking starts from the beginning
        bpy.ops.screen.frame_jump(end=False)

        sorted_objects = sorted(selected_objects, key=lambda o: o.name)
        original_active = context.view_layer.objects.active

        for i, obj in enumerate(sorted_objects):
            context.view_layer.objects.active = obj
            l = h
            h = round(spectrum_Start * (base ** (i + 1)), 2)
            print(f"     Animating: {obj.name}   l: {l} Hz    h: {h} Hz")

            bake_sound_to_channels(
                obj, context, path, l, h, intensity,
                sav_props.animate_location,
                sav_props.animate_rotation,
                sav_props.animate_scale,
                sav_props.animate_axis_x,
                sav_props.animate_axis_y,
                sav_props.animate_axis_z,
                sav_props.additive_mode)

            done_percent = (i + 1) / columns * 100
            elapsed_minutes = (time.time() - start_time) / 60
            print(f"{done_percent:.1f}%   done in {elapsed_minutes:.2f} minutes")

        context.view_layer.objects.active = original_active
        print(f"FINISHED Animating {columns} objects")

        return {'FINISHED'}


class SimpleAudioVisualizerPanel(bpy.types.Panel):
    bl_label = "Simple Audio Visualizer"
    bl_idname = "GRAPH_EDITOR_PT_simple_audio_visualizer"
    bl_space_type = 'GRAPH_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Simple Audio Visualizer"

    def draw(self, context):
        layout = self.layout
        sav_props = context.scene.sav_props

        # Global settings
        layout.prop(sav_props, "input_mode")
        if sav_props.input_mode == 'FILE':
            layout.prop(sav_props, "path")
        else:
            if not _SAV_HAVE_SD:
                box = layout.box()
                box.label(text="Install 'sounddevice' in Blender's Python", icon='ERROR')
            if sav_props.input_mode == 'MIC':
                layout.prop(sav_props, "mic_device")
            elif sav_props.input_mode == 'SPEAKERS':
                layout.prop(sav_props, "spk_device")
            row = layout.row(align=True)
            row.prop(sav_props, "live_update_interval")
            row.prop(sav_props, "fft_size")
            layout.prop(sav_props, "smoothing")
            layout.prop(sav_props, "reset_on_silence")
            if sav_props.reset_on_silence:
                row = layout.row(align=True)
                row.prop(sav_props, "silence_threshold")
                row.prop(sav_props, "silence_hold")
        layout.prop(sav_props, "intensity")
        layout.separator()

        # Box 1: Create new objects
        box_create = layout.box()
        row = box_create.row()
        row.label(text="Generate New Objects")
        row.alignment = 'RIGHT'
        row.prop(sav_props, "info_generate", text="", icon='QUESTION', emboss=False)
        box_create.prop(sav_props, "columns")
        box_create.prop(sav_props, "mirror_mode")
        box_create.operator("object.simple_audio_visualizer", text="Create Visualizer")

        # Box 2: Animate selected objects
        box_visualize = layout.box()
        row = box_visualize.row()
        row.label(text="Animate Selected Objects")
        row.alignment = 'RIGHT'
        row.prop(sav_props, "info_animate", text="", icon='QUESTION', emboss=False)

        # Sub-box for toggles
        sub_box = box_visualize.box()
        col = sub_box.column(align=True)

        row = col.row(align=True)
        row.prop(sav_props, "animate_location", toggle=True)
        row.prop(sav_props, "animate_rotation", toggle=True)
        row.prop(sav_props, "animate_scale", toggle=True)

        row = col.row(align=True)
        row.prop(sav_props, "animate_axis_x", toggle=True)
        row.prop(sav_props, "animate_axis_y", toggle=True)
        row.prop(sav_props, "animate_axis_z", toggle=True)

        box_visualize.prop(sav_props, "additive_mode")

        row = box_visualize.row()
        row.enabled = bool(context.selected_objects)
        row.operator("object.visualize_selected_objects", text="Visualize Selected Objects")

        # Live controls
        if sav_props.input_mode != 'FILE':
            layout.separator()
            # Status indicator
            if sav_props.live_active:
                layout.label(text=f"Live: ACTIVE  {sav_props.live_device_label}", icon='PLAY')
            else:
                layout.label(text="Live: Inactive", icon='PAUSE')
            if sav_props.live_status:
                layout.label(text=sav_props.live_status, icon='INFO')
            # Routing options
            layout.prop(sav_props, "live_route")
            if sav_props.live_route == 'GN_BANDS':
                layout.prop(sav_props, "gn_output_mode")
                row = layout.row(align=True)
                row.prop(sav_props, "gn_bands")
                if sav_props.gn_output_mode == 'SOCKETS':
                    row.prop(sav_props, "gn_collection_hide")
                    layout.prop(sav_props, "gn_output_type")
                layout.operator("object.sav_setup_bands_group", text="Create/Update Bands Node Group", icon='NODETREE')
            live_row = layout.row(align=True)
            # For GN_BANDS, selection is not required
            live_row.enabled = _SAV_HAVE_SD and (bool(context.selected_objects) or sav_props.live_route == 'GN_BANDS')
            live_row.operator("object.sav_start_live", text="Start Live", icon='PLAY')
            live_row.operator("object.sav_stop_live", text="Stop Live", icon='PAUSE')

# -----------------------------
# Live audio helpers/operators
# -----------------------------

def _sav_make_wasapi_loopback_settings():
    try:
        return sd.WasapiSettings(loopback=True)
    except TypeError:
        ws = sd.WasapiSettings()
        if hasattr(ws, 'loopback'):
            ws.loopback = True
            return ws
        raise RuntimeError("This sounddevice build lacks WASAPI loopback support.")

def _sav_audio_callback(indata, frames, time_info, status):
    global _sav_live_last_chunk, _sav_audio_ring, _sav_audio_ring_len
    if status:
        # Status may contain xruns; still capture
        pass
    try:
        # Copy to avoid re-use of the buffer by PortAudio
        arr = np.array(indata, dtype=np.float32, copy=True)
        _sav_live_last_chunk = arr
        # Append mono samples to ring buffer
        if arr.ndim == 2 and arr.shape[1] > 1:
            mono = np.mean(arr, axis=1).astype(np.float32, copy=False)
        elif arr.ndim == 2:
            mono = arr[:, 0].astype(np.float32, copy=False)
        else:
            mono = arr.reshape(-1).astype(np.float32, copy=False)
        _sav_audio_ring.append(mono)
        _sav_audio_ring_len += mono.shape[0]
        # Trim ring to max samples
        while _sav_audio_ring_len > _SAV_MAX_RING_SAMPLES and len(_sav_audio_ring) > 1:
            popped = _sav_audio_ring.popleft()
            _sav_audio_ring_len -= popped.shape[0]
    except Exception:
        _sav_live_last_chunk = None

def _sav_ensure_bands_assets(context, bands, hide_controllers):
    """Ensure controller empties and a GN node group exist for 'bands' count.
    Returns a list of empty objects in band order.
    """
    # 1) Ensure controller empties
    coll_name = "SAV Bands Controllers"
    coll = bpy.data.collections.get(coll_name)
    if coll is None:
        coll = bpy.data.collections.new(coll_name)
        if context.scene.collection:
            context.scene.collection.children.link(coll)
    empties = []
    desired = [f"SAV_Band_{i+1:02d}" for i in range(int(bands))]
    # Create missing empties
    for name in desired:
        obj = bpy.data.objects.get(name)
        if obj is None:
            obj = bpy.data.objects.new(name, None)
            obj.empty_display_type = 'PLAIN_AXES'
            obj.empty_display_size = 0.05
            coll.objects.link(obj)
        # Make sure it's in the controllers collection
        if obj.name not in coll.objects:
            try:
                coll.objects.link(obj)
            except Exception:
                pass
        # Hide if requested
        obj.hide_viewport = bool(hide_controllers)
        obj.hide_render = bool(hide_controllers)
        empties.append(obj)
    # Remove extra old empties with the pattern if any
    existing = [o for o in coll.objects if o.name.startswith("SAV_Band_")]
    for obj in existing:
        if obj.name not in desired:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass

    # 2) Ensure a GN node group that exposes outputs for each band
    _sav_create_or_update_bands_group(empties, context.scene.sav_props.gn_output_type)
    empties.sort(key=lambda o: o.name)
    return empties

def _sav_ensure_points_assets(context, bands):
    """Ensure a points mesh object exists with 'bands' vertices, arranged along X.
    Returns the object.
    """
    name = "SAV_Bands_Geometry"
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != 'MESH':
        me = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, me)
        context.scene.collection.objects.link(obj)
    me = obj.data
    # Rebuild geometry if counts differ
    if len(me.vertices) != int(bands):
        try:
            me.clear_geometry()
        except Exception:
            pass
        verts = [(float(i), 0.0, 0.0) for i in range(int(bands))]
        me.from_pydata(verts, [], [])
        me.update()
    # View settings
    obj.hide_render = True
    return obj

def _sav_create_or_update_points_group(points_obj):
    """Create or update a GN group that outputs the points geometry as a single socket."""
    group_name = "SAV Bands Points"
    nt = bpy.data.node_groups.get(group_name)
    if nt is None:
        nt = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
    try:
        nt.nodes.clear()
    except Exception:
        pass
    nodes = nt.nodes
    links = nt.links
    out = nodes.new('NodeGroupOutput')
    out.location = (400, 0)
    # Interface: single geometry output "Bands"
    try:
        for sock in list(nt.interface.items_tree):
            if getattr(sock, 'in_out', 'OUTPUT') == 'OUTPUT':
                nt.interface.remove(sock)
    except Exception:
        try:
            for s in list(nt.outputs):
                nt.outputs.remove(s)
        except Exception:
            pass
    try:
        nt.interface.new_socket(name='Bands', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    except Exception:
        try:
            nt.outputs.new('NodeSocketGeometry', 'Bands')
        except Exception:
            pass
    n = nodes.new('GeometryNodeObjectInfo')
    n.location = (0, 0)
    n.inputs[0].default_value = points_obj
    n.as_instance = False if hasattr(n, 'as_instance') else False
    try:
        links.new(n.outputs['Geometry'], out.inputs['Bands'])
    except Exception:
        try:
            links.new(n.outputs[0], out.inputs[0])
        except Exception:
            pass

def _sav_create_or_update_bands_group(empties, output_type):
    """Create or update a Geometry Nodes group with one Vector output per empty.
    Each output is driven by an Object Info node reading the corresponding empty's location.
    """
    group_name = "SAV Bands Outputs"
    nt = bpy.data.node_groups.get(group_name)
    if nt is None:
        nt = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
    # Clear nodes for a clean rebuild
    try:
        nt.nodes.clear()
    except Exception:
        pass
    nodes = nt.nodes
    links = nt.links
    out = nodes.new('NodeGroupOutput')
    out.location = (500, 0)
    # Clear and rebuild interface outputs depending on Blender version
    def _clear_outputs(ntree):
        try:
            # Blender 4.x interface API
            for sock in list(ntree.interface.items_tree):
                # Remove only outputs; inputs we don't use
                if getattr(sock, 'in_out', 'OUTPUT') == 'OUTPUT':
                    ntree.interface.remove(sock)
        except Exception:
            try:
                # Older API
                for s in list(ntree.outputs):
                    ntree.outputs.remove(s)
            except Exception:
                pass
    def _add_output(ntree, name):
        try:
            sock_type = (
                'NodeSocketVector' if output_type == 'VECTOR'
                else ('NodeSocketFloat' if output_type == 'FLOAT' else 'NodeSocketInt')
            )
            ntree.interface.new_socket(name=name, in_out='OUTPUT', socket_type=sock_type)
        except Exception:
            try:
                alt_type = (
                    'NodeSocketVector' if output_type == 'VECTOR'
                    else ('NodeSocketFloat' if output_type == 'FLOAT' else 'NodeSocketInt')
                )
                ntree.outputs.new(alt_type, name)
            except Exception:
                pass
    def _add_index_output(ntree, name):
        # Always Float for index
        try:
            ntree.interface.new_socket(name=name, in_out='OUTPUT', socket_type='NodeSocketFloat')
        except Exception:
            try:
                ntree.outputs.new('NodeSocketFloat', name)
            except Exception:
                pass
    _clear_outputs(nt)
    # Add one output per empty and object info feeding it
    y = 200 * (len(empties) - 1)
    for i, obj in enumerate(sorted(empties, key=lambda o: o.name)):
        label = f"Band {i+1:02d}"
        _add_output(nt, label)
        idx_label = f"{label} Index"
        _add_index_output(nt, idx_label)
        n = nodes.new('GeometryNodeObjectInfo')
        n.location = (0, -i * 200 + y)
        n.inputs[0].default_value = obj  # Object input
        n.transform_space = 'RELATIVE'
        # Link depending on output type
        if output_type == 'VECTOR':
            try:
                links.new(n.outputs['Location'], out.inputs[label])
            except Exception:
                try:
                    links.new(n.outputs[1], out.inputs[i])
                except Exception:
                    pass
        else:
            # FLOAT or INT: separate XYZ and use X component
            sep = nodes.new('ShaderNodeSeparateXYZ')
            sep.location = (200, -i * 200 + y)
            try:
                links.new(n.outputs['Location'], sep.inputs['Vector'])
            except Exception:
                try:
                    links.new(n.outputs[1], sep.inputs[0])
                except Exception:
                    pass
            if output_type == 'INT':
                # Add a Math node set to ROUND for integral output
                try:
                    m = nodes.new('ShaderNodeMath')
                    m.operation = 'ROUND'
                    m.location = (380, -i * 200 + y)
                    try:
                        links.new(sep.outputs['X'], m.inputs[0])
                    except Exception:
                        links.new(sep.outputs[0], m.inputs[0])
                    try:
                        links.new(m.outputs['Value'], out.inputs[label])
                    except Exception:
                        links.new(m.outputs[0], out.inputs[i])
                except Exception:
                    # Fallback: direct link if node type unavailable
                    try:
                        links.new(sep.outputs['X'], out.inputs[label])
                    except Exception:
                        try:
                            links.new(sep.outputs[0], out.inputs[i])
                        except Exception:
                            pass
            else:
                # FLOAT: direct link
                try:
                    links.new(sep.outputs['X'], out.inputs[label])
                except Exception:
                    try:
                        links.new(sep.outputs[0], out.inputs[i])
                    except Exception:
                        pass
        # Create and link the index (1-based)
        try:
            val = nodes.new('ShaderNodeValue')
            val.outputs[0].default_value = float(i + 1)
            val.location = (380, -i * 200 + y + 60)
            try:
                links.new(val.outputs['Value'], out.inputs[idx_label])
            except Exception:
                # Find the socket by name or fallback to scanning
                try:
                    links.new(val.outputs[0], out.inputs[idx_label])
                except Exception:
                    # As a last resort, assume index socket is right after the band socket
                    band_index = list(out.inputs.keys()).index(label) if hasattr(out.inputs, 'keys') else i
                    links.new(val.outputs[0], out.inputs[band_index + 1])
        except Exception:
            pass

def _sav_build_targets(context):
    """Prepare targets and frequency edges for live routing."""
    global _sav_live_targets, _sav_live_freq_edges, _sav_live_prev_vals
    s = context.scene.sav_props
    # If routing to GN bands, ensure assets and target empties, otherwise use selection
    if s.live_route == 'GN_BANDS':
        if getattr(s, 'gn_output_mode', 'SOCKETS') == 'POINTS':
            points_obj = _sav_ensure_points_assets(context, s.gn_bands)
            _sav_create_or_update_points_group(points_obj)
            # Synthesize placeholder targets just to size arrays; live tick will handle mesh updates
            sel = [points_obj] * int(s.gn_bands)
        else:
            empties = _sav_ensure_bands_assets(context, s.gn_bands, s.gn_collection_hide)
            sel = empties
    else:
        sel = sorted(context.selected_objects, key=lambda o: o.name)
        if not sel:
            return False

    # Determine which channels and axes are active; mirror existing behavior by mapping per-object to Z scale by default
    # Build target list: one band per selected object
    _sav_live_targets = []
    for obj in sel:
        if s.live_route == 'GN_BANDS' and getattr(s, 'gn_output_mode', 'SOCKETS') == 'SOCKETS':
            # Drive all XYZ of empty location equally as a vector output
            data_path = 'location'
            axis_index = 0  # unused; we'll set all axes
            base_val = 0.0
            _sav_live_targets.append({'name': obj.name, 'path': data_path, 'axis': axis_index, 'baseline': base_val, 'all_axes': True})
        else:
            # Default: scale Z
            data_path = 'scale'
            axis_index = 2
            # If the user chose other options, honor them
            if s.animate_location:
                data_path = 'location'
            elif s.animate_rotation:
                data_path = 'rotation_euler'
            elif s.animate_scale:
                data_path = 'scale'
            # Axis preference
            if s.animate_axis_x:
                axis_index = 0
            elif s.animate_axis_y:
                axis_index = 1
            else:
                axis_index = 2
            # Capture baseline value for this object's channel/axis
            try:
                ch = getattr(obj, data_path)
                base_val = float(ch[axis_index])
            except Exception:
                base_val = 0.0 if data_path != 'scale' else 1.0
            _sav_live_targets.append({'name': obj.name, 'path': data_path, 'axis': axis_index, 'baseline': base_val})

    # Compute log-spaced frequency edges
    n = len(_sav_live_targets)
    f_lo = 10.0
    f_hi = 21000.0
    base = (f_hi / f_lo) ** (1.0 / n)
    edges = []
    l = 1.0
    h = f_lo
    for i in range(n):
        l = h
        h = round(f_lo * (base ** (i + 1)), 2)
        edges.append((l, h))
    _sav_live_freq_edges = edges
    _sav_live_prev_vals = np.zeros((n,), dtype=np.float32)
    return True

def _sav_apply_value(obj, path, axis, value):
    try:
        channel = getattr(obj, path)
        if isinstance(channel, (list, tuple)):
            vec = list(channel)
            vec[axis] = value
            setattr(obj, path, type(channel)(vec))
        else:
            channel[axis] = value
    except Exception:
        # Best effort
        try:
            setattr(obj, path, value)
        except Exception:
            pass

def _sav_live_timer():
    global _sav_live_should_run, _sav_live_last_chunk, _sav_live_prev_vals, _sav_audio_ring, _sav_audio_ring_len, _sav_silence_last_loud_time
    if not _sav_live_should_run:
        return None
    s = bpy.context.scene.sav_props
    # Build a mono signal from ring buffer
    if _sav_audio_ring_len <= 0:
        return s.live_update_interval
    try:
        # Gather latest samples
        if len(_sav_audio_ring) == 1:
            sig = _sav_audio_ring[0]
        else:
            sig = np.concatenate(list(_sav_audio_ring), dtype=np.float32)
        n = int(s.fft_size)
        n = max(256, min(16384, 1 << int(round(math.log2(n)))))
        if sig.shape[0] < n:
            return s.live_update_interval
        sig = sig[-n:]

        # RMS for silence detection
        rms = float(np.sqrt(np.mean(sig.astype(np.float32) ** 2) + 1e-12))

        # FFT
        window = np.hanning(n).astype(np.float32)
        spec = np.abs(np.fft.rfft(sig * window))
        freqs = np.fft.rfftfreq(n, d=1.0 / float(_sav_live_samplerate))

        # Map spectrum to targets
        out_vals = []
        for (lo, hi) in _sav_live_freq_edges:
            mask = (freqs >= lo) & (freqs < hi)
            energy = float(np.mean(spec[mask])) if np.any(mask) else 0.0
            out_vals.append(energy)
        out_vals = np.array(out_vals, dtype=np.float32)

        # Normalize and scale
        if out_vals.size and out_vals.max() > 1e-8:
            out_vals = out_vals / (out_vals.max() + 1e-8)
        out_vals *= float(s.intensity)
        out_vals = np.maximum(out_vals, 0.0)

        # Smooth
        a = float(s.smoothing)
        if _sav_live_prev_vals is None or len(_sav_live_prev_vals) != len(out_vals):
            _sav_live_prev_vals = np.zeros_like(out_vals, dtype=np.float32)
        out_vals = a * _sav_live_prev_vals + (1.0 - a) * out_vals
        _sav_live_prev_vals = out_vals

        # Silence handling
        now = time.time()
        if s.reset_on_silence:
            if rms >= float(s.silence_threshold):
                _sav_silence_last_loud_time = now
            elif (now - _sav_silence_last_loud_time) >= float(s.silence_hold):
                missing = 0
                for t in _sav_live_targets:
                    obj = bpy.data.objects.get(t['name'])
                    if obj is None:
                        missing += 1
                        continue
                    try:
                        path = t['path']; axis = t['axis']; base_val = float(t.get('baseline', 1.0 if t['path']=='scale' else 0.0))
                        if path == 'scale':
                            base = list(obj.scale); base[axis] = base_val; obj.scale = type(obj.scale)(base)
                        elif path == 'location':
                            base = list(obj.location); base[axis] = base_val; obj.location = type(obj.location)(base)
                        elif path == 'rotation_euler':
                            base = list(obj.rotation_euler); base[axis] = base_val; obj.rotation_euler = type(obj.rotation_euler)(base)
                    except Exception:
                        pass
                if _sav_live_prev_vals is not None:
                    _sav_live_prev_vals = np.zeros_like(_sav_live_prev_vals)
                try:
                    s.live_status = f"Silence reset • rms={rms:.4f}"
                except Exception:
                    pass
                if missing == len(_sav_live_targets):
                    _sav_live_should_run = False
                    return None
                return s.live_update_interval

        # Apply to targets
        # If routing to points geometry, update the mesh vertex positions (z = amplitude)
        if s.live_route == 'GN_BANDS' and getattr(s, 'gn_output_mode', 'SOCKETS') == 'POINTS':
            obj = bpy.data.objects.get('SAV_Bands_Geometry')
            if obj is None or obj.type != 'MESH' or len(obj.data.vertices) != len(out_vals):
                try:
                    obj = _sav_ensure_points_assets(bpy.context, len(out_vals))
                    _sav_create_or_update_points_group(obj)
                except Exception:
                    return s.live_update_interval
            me = obj.data
            try:
                for i in range(len(out_vals)):
                    co = me.vertices[i].co
                    me.vertices[i].co = (co.x, co.y, float(out_vals[i]))
                me.update()
            except Exception:
                pass
            try:
                peak = float(np.max(out_vals)) if out_vals.size else 0.0
                s.live_status = f"Live tick ok • points={len(out_vals)} • peak={peak:.2f} • rms={rms:.3f}"
            except Exception:
                pass
            return s.live_update_interval

        missing = 0
        for i, t in enumerate(_sav_live_targets):
            name = t['name']; path = t['path']; axis = t['axis']
            baseline = float(t.get('baseline', 1.0 if path == 'scale' else 0.0))
            obj = bpy.data.objects.get(name)
            if obj is None:
                missing += 1
                continue
            try:
                if path == 'scale':
                    base = list(obj.scale)
                    base[axis] = (baseline + out_vals[i]) if not s.additive_mode else (base[axis] + out_vals[i])
                    obj.scale = type(obj.scale)(base)
                elif path == 'location':
                    base = list(obj.location)
                    if t.get('all_axes', False):
                        val = (base[0] + out_vals[i]) if s.additive_mode else (baseline + out_vals[i])
                        base = [val, val, val]
                    else:
                        base[axis] = (base[axis] + out_vals[i]) if s.additive_mode else (baseline + out_vals[i])
                    obj.location = type(obj.location)(base)
                elif path == 'rotation_euler':
                    base = list(obj.rotation_euler)
                    base[axis] = (base[axis] + out_vals[i]) if s.additive_mode else (baseline + out_vals[i])
                    obj.rotation_euler = type(obj.rotation_euler)(base)
            except Exception:
                pass
        if missing == len(_sav_live_targets):
            _sav_live_should_run = False
            return None

        # Status
        try:
            peak = float(np.max(out_vals)) if out_vals.size else 0.0
            s.live_status = f"Live tick ok • bands={len(_sav_live_targets)} • peak={peak:.2f} • rms={rms:.3f}"
        except Exception:
            pass
    except Exception as e:
        print(f"SAV live update error: {e}")
    return s.live_update_interval


class SAV_OT_StartLive(bpy.types.Operator):
    bl_idname = "object.sav_start_live"
    bl_label = "Start Live"
    bl_description = "Start live audio visualization from mic or speakers"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global _sav_live_stream, _sav_live_should_run, _sav_live_samplerate, _sav_silence_last_loud_time
        s = context.scene.sav_props
        if _sav_live_should_run:
            msg = "Live mode is already running"
            print(f"[SAV] {msg}")
            self.report({'INFO'}, msg)
            s.live_active = True
            return {'FINISHED'}
        if not _SAV_HAVE_SD:
            self.report({'ERROR'}, "sounddevice is not available in Blender's Python")
            return {'CANCELLED'}
        if s.input_mode not in {'MIC', 'SPEAKERS'}:
            self.report({'ERROR'}, "Switch Input Source to MIC or SPEAKERS for live mode")
            return {'CANCELLED'}
        if not _sav_build_targets(context):
            self.report({'ERROR'}, "Select objects to drive before starting live mode")
            return {'CANCELLED'}

        device_index = None
        extra = None
        try:
            if s.input_mode == 'MIC':
                device_index = int(s.mic_device) if s.mic_device else None
            else:
                # Speakers loopback (WASAPI only)
                device_index = int(s.spk_device) if s.spk_device else None
                extra = _sav_make_wasapi_loopback_settings()
        except Exception as e:
            self.report({'ERROR'}, f"Device selection error: {e}")
            return {'CANCELLED'}

        if device_index is None:
            self.report({'ERROR'}, "Please choose a valid device for live input")
            return {'CANCELLED'}

        # Determine channels and sample rate
        dev = sd.query_devices(device_index)
        ch = max(1, min(2, int(dev.get('max_input_channels' if s.input_mode=='MIC' else 'max_output_channels', 1))))
        _sav_live_samplerate = int(sd.query_devices(device_index)['default_samplerate'] or 48000)

        try:
            _sav_live_stream = sd.InputStream(
                device=device_index,
                channels=ch,
                samplerate=_sav_live_samplerate,
                dtype='float32',
                callback=_sav_audio_callback,
                extra_settings=extra,
                blocksize=int(s.fft_size // 2) if int(s.fft_size) >= 512 else 256,
                latency='low'
            )
            _sav_live_stream.start()
            _sav_live_should_run = True
            # Reset ring buffer
            global _sav_audio_ring, _sav_audio_ring_len
            _sav_audio_ring = deque()
            _sav_audio_ring_len = 0
            _sav_silence_last_loud_time = time.time()
            bpy.app.timers.register(_sav_live_timer)
            # UI + console status
            try:
                ha = sd.query_hostapis()[sd.query_devices(device_index)['hostapi']]['name']
            except Exception:
                ha = ''
            dev_name = sd.query_devices(device_index)['name']
            mode = s.input_mode
            s.live_active = True
            s.live_device_label = f"{dev_name} ({ha}) @ {_sav_live_samplerate}Hz, ch={ch}, FFT={s.fft_size}"
            s.live_status = f"Started live mode: {mode}"
            print(f"[SAV] Live started: {mode} -> {s.live_device_label}")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start live audio: {e}")
            print(f"[SAV] Live start FAILED: {e}")
            s.live_active = False
            s.live_status = f"Start failed: {e}"
            return {'CANCELLED'}

        return {'FINISHED'}


class SAV_OT_StopLive(bpy.types.Operator):
    bl_idname = "object.sav_stop_live"
    bl_label = "Stop Live"
    bl_description = "Stop live audio visualization"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global _sav_live_stream, _sav_live_should_run
        _sav_live_should_run = False
        s = context.scene.sav_props
        try:
            if _sav_live_stream:
                _sav_live_stream.stop()
                _sav_live_stream.close()
        finally:
            _sav_live_stream = None
        # Clear ring buffer when stopping
        global _sav_audio_ring, _sav_audio_ring_len
        _sav_audio_ring = deque()
        _sav_audio_ring_len = 0
        s.live_active = False
        s.live_status = "Stopped"
        s.live_device_label = ""
        print("[SAV] Live stopped")
        return {'FINISHED'}


class SAV_OT_SetupBandsGroup(bpy.types.Operator):
    bl_idname = "object.sav_setup_bands_group"
    bl_label = "Setup Bands Node Group"
    bl_description = "Create or update a Geometry Nodes group with vector outputs for each band, driven by controller empties"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.sav_props
        try:
            if getattr(s, 'gn_output_mode', 'SOCKETS') == 'POINTS':
                points_obj = _sav_ensure_points_assets(context, s.gn_bands)
                _sav_create_or_update_points_group(points_obj)
                self.report({'INFO'}, f"Bands points ready: {int(s.gn_bands)} points")
                s.live_status = f"Bands points ready: {int(s.gn_bands)} points"
            else:
                empties = _sav_ensure_bands_assets(context, s.gn_bands, s.gn_collection_hide)
                self.report({'INFO'}, f"Bands group ready: {len(empties)} outputs")
                s.live_status = f"Bands group ready: {len(empties)} outputs"
        except Exception as e:
            self.report({'ERROR'}, f"Failed to setup bands group: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def register():
    bpy.utils.register_class(SimpleAudioVisualizerProperties)
    bpy.types.Scene.sav_props = bpy.props.PointerProperty(type=SimpleAudioVisualizerProperties)

    bpy.utils.register_class(SimpleAudioVisualizerOperator)
    bpy.utils.register_class(VisualizeSelectedObjectsOperator)

    bpy.utils.register_class(SimpleAudioVisualizerPanel)
    bpy.utils.register_class(SAV_OT_StartLive)
    bpy.utils.register_class(SAV_OT_StopLive)
    bpy.utils.register_class(SAV_OT_SetupBandsGroup)


def unregister():
    # Ensure live resources are cleaned up on unload
    global _sav_live_stream, _sav_live_should_run
    try:
        _sav_live_should_run = False
        if _sav_live_stream:
            _sav_live_stream.stop()
            _sav_live_stream.close()
    except Exception:
        pass
    finally:
        _sav_live_stream = None
    bpy.utils.unregister_class(SimpleAudioVisualizerPanel)
    bpy.utils.unregister_class(SimpleAudioVisualizerOperator)
    bpy.utils.unregister_class(VisualizeSelectedObjectsOperator)
    bpy.utils.unregister_class(SAV_OT_StopLive)
    bpy.utils.unregister_class(SAV_OT_StartLive)
    bpy.utils.unregister_class(SAV_OT_SetupBandsGroup)

    del bpy.types.Scene.sav_props
    bpy.utils.unregister_class(SimpleAudioVisualizerProperties)
