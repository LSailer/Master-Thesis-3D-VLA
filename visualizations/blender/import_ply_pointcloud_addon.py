"""Import PLY files as colored point clouds, ready to view.

Adds File > Import > PLY as Colored Point Cloud (.ply).

For each selected .ply file it:
  1. imports it with Blender's built-in PLY importer,
  2. converts the vertex-only mesh to a Point Cloud object,
  3. sets a per-point radius,
  4. assigns a shared material that reads the 'Col' color attribute
     (created once, reused for every import),
  5. adds a shared Geometry Nodes modifier that instances smooth
     spheres on the points, so points look round in EEVEE too
     (EEVEE's native point display uses low-poly instances that
     look like squares up close; Cycles alone renders true spheres),
  6. switches the 3D viewport to Material Preview so colors are
     visible, and lowers Clip Start so close-up inspection works.

Install: Edit > Preferences > Add-ons > arrow menu (top right) >
Install from Disk... > pick this file > enable the checkbox.

Tested with Blender 5.2 LTS; needs 4.2+.
"""

bl_info = {
    "name": "Import PLY as Colored Point Cloud",
    "author": "Luca Sailer + Claude",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "File > Import > PLY as Colored Point Cloud (.ply)",
    "description": "Import XYZRGB PLY point clouds with colors visible",
    "category": "Import-Export",
}

import os

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    FloatProperty,
    StringProperty,
)
from bpy_extras.io_utils import ImportHelper

MATERIAL_NAME = "PointCloudRGB"
NODEGROUP_NAME = "PointCloudSpheres"
COLOR_ATTRIBUTE = "Col"  # name Blender's PLY importer gives red/green/blue


MATERIAL_VERSION = 2  # bump when _build_material_nodes changes


def _build_material_nodes(mat: bpy.types.Material) -> None:
    """(Re)build the point-cloud material node tree.

    The color is read twice and summed: 'Geometry' resolves 'Col' on a
    native point cloud, 'Instancer' resolves it on the spheres instanced
    by the Round Points modifier. Whichever path is inactive yields
    black, so the sum is always the true point color.
    """
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    attr_geo = nt.nodes.new("ShaderNodeAttribute")
    attr_geo.attribute_name = COLOR_ATTRIBUTE
    attr_geo.attribute_type = "GEOMETRY"
    attr_geo.location = (-600, 100)
    attr_inst = nt.nodes.new("ShaderNodeAttribute")
    attr_inst.attribute_name = COLOR_ATTRIBUTE
    attr_inst.attribute_type = "INSTANCER"
    attr_inst.location = (-600, -150)

    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "ADD"
    mix.inputs["Factor"].default_value = 1.0
    mix.location = (-300, 0)
    nt.links.new(attr_geo.outputs["Color"], mix.inputs["A"])
    nt.links.new(attr_inst.outputs["Color"], mix.inputs["B"])

    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    # Emission makes the raw RGB visible independent of scene lighting.
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.0
    mat["pc_addon_material_version"] = MATERIAL_VERSION


def _get_or_create_material() -> bpy.types.Material:
    """Return the shared point-cloud material, creating/upgrading it."""
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if mat is None:
        mat = bpy.data.materials.new(MATERIAL_NAME)
    if mat.get("pc_addon_material_version", 0) < MATERIAL_VERSION:
        _build_material_nodes(mat)
    return mat


def _get_or_create_sphere_nodegroup() -> bpy.types.GeometryNodeTree:
    """Return the shared points-to-spheres node group, creating it once.

    EEVEE draws point clouds as very low-poly sphere instances whose
    silhouette looks square/diamond shaped up close. Instancing our own
    smooth-shaded ico spheres (scaled by the 'radius' attribute) gives
    round points in every engine.
    """
    ng = bpy.data.node_groups.get(NODEGROUP_NAME)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(NODEGROUP_NAME, "GeometryNodeTree")
    ng.is_modifier = True
    ng.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    subdiv_sock = ng.interface.new_socket(
        name="Sphere Subdivisions",
        in_out="INPUT",
        socket_type="NodeSocketInt",
    )
    subdiv_sock.default_value = 2
    subdiv_sock.min_value = 1
    subdiv_sock.max_value = 4
    ng.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    gin = ng.nodes.new("NodeGroupInput")
    gin.location = (-500, 0)
    gout = ng.nodes.new("NodeGroupOutput")
    gout.location = (300, 0)

    ico = ng.nodes.new("GeometryNodeMeshIcoSphere")
    ico.location = (-500, -200)
    ico.inputs["Radius"].default_value = 1.0

    smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
    smooth.location = (-320, -200)

    setmat = ng.nodes.new("GeometryNodeSetMaterial")
    setmat.location = (-140, -200)
    setmat.inputs["Material"].default_value = _get_or_create_material()

    radius_attr = ng.nodes.new("GeometryNodeInputNamedAttribute")
    radius_attr.location = (-500, -420)
    radius_attr.data_type = "FLOAT"
    radius_attr.inputs["Name"].default_value = "radius"

    inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
    inst.location = (100, 0)

    ng.links.new(gin.outputs["Geometry"], inst.inputs["Points"])
    ng.links.new(gin.outputs["Sphere Subdivisions"], ico.inputs["Subdivisions"])
    ng.links.new(ico.outputs["Mesh"], smooth.inputs["Geometry"])
    ng.links.new(smooth.outputs["Geometry"], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs["Geometry"], inst.inputs["Instance"])
    ng.links.new(radius_attr.outputs["Attribute"], inst.inputs["Scale"])
    ng.links.new(inst.outputs["Instances"], gout.inputs["Geometry"])
    return ng


def _set_radius(pointcloud, radius: float) -> None:
    if "radius" in pointcloud.attributes:
        attr = pointcloud.attributes["radius"]
    else:
        attr = pointcloud.attributes.new(
            name="radius", type="FLOAT", domain="POINT"
        )
    n = len(pointcloud.points)
    attr.data.foreach_set("value", [radius] * n)


class IMPORT_OT_ply_pointcloud(bpy.types.Operator, ImportHelper):
    """Import one or more PLY files as colored point clouds."""

    bl_idname = "import_scene.ply_pointcloud"
    bl_label = "Import PLY as Colored Point Cloud"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement)
    directory: StringProperty(subtype="DIR_PATH")

    point_radius: FloatProperty(
        name="Point Radius",
        description="Display radius per point, in meters",
        default=0.005,
        min=0.0,
        soft_max=0.1,
        precision=4,
    )

    round_points: BoolProperty(
        name="Round Points",
        description=(
            "Instance smooth spheres on the points so they look round "
            "in EEVEE/Material Preview (EEVEE's native point display "
            "looks square up close). Disable to keep native points"
        ),
        default=True,
    )

    def execute(self, context):
        paths = [
            os.path.join(self.directory, f.name) for f in self.files if f.name
        ]
        if not paths and self.filepath:
            paths = [self.filepath]
        if not paths:
            self.report({"ERROR"}, "No .ply file selected")
            return {"CANCELLED"}

        mat = _get_or_create_material()
        imported = []
        for path in paths:
            bpy.ops.wm.ply_import(filepath=path)
            obj = context.active_object
            if obj is None:
                self.report({"WARNING"}, f"Import failed: {path}")
                continue
            if obj.type == "MESH":
                bpy.ops.object.convert(target="POINTCLOUD")
                obj = context.active_object
            pc = obj.data
            _set_radius(pc, self.point_radius)
            if COLOR_ATTRIBUTE not in pc.attributes:
                self.report(
                    {"WARNING"},
                    f"{os.path.basename(path)}: no '{COLOR_ATTRIBUTE}' "
                    "attribute — file has no red/green/blue properties; "
                    "points will render white",
                )
            if not pc.materials:
                pc.materials.append(mat)
            if self.round_points and not any(
                m.type == "NODES" and m.node_group is not None
                and m.node_group.name == NODEGROUP_NAME
                for m in obj.modifiers
            ):
                mod = obj.modifiers.new("Round Points", "NODES")
                mod.node_group = _get_or_create_sphere_nodegroup()
            imported.append(obj.name)

        # Colors only show in Material Preview / Rendered shading. Also
        # lower Clip Start so zooming close to points doesn't clip them.
        screen = getattr(context, "screen", None)
        for area in screen.areas if screen else []:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                if space.shading.type == "SOLID":
                    space.shading.type = "MATERIAL"
                space.clip_start = min(space.clip_start, 0.001)

        self.report(
            {"INFO"}, f"Imported {len(imported)} colored point cloud(s)"
        )
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(
        IMPORT_OT_ply_pointcloud.bl_idname,
        text="PLY as Colored Point Cloud (.ply)",
    )


def register():
    bpy.utils.register_class(IMPORT_OT_ply_pointcloud)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(IMPORT_OT_ply_pointcloud)


if __name__ == "__main__":
    register()
