# Viewing PLY point clouds in Blender

The reconstruction point clouds (e.g. `output/3d_pointclouds/*.ply`,
`output/methods/scenes/3d_pointclouds/*.ply`) are XYZRGB PLY files. Blender's
stock PLY importer loads them as vertex-only meshes with no visible color; the
add-on in this folder fixes that in one click.

## Quick start (recommended): the add-on

One-time install of `import_ply_pointcloud_addon.py` (ships next to this
README), then every import is one click. Verified end-to-end on
`output/3d_pointclouds/VGGT_pointcloud.ply`: import → Point Cloud with `Col`,
`radius`, and the `PointCloudRGB` material, colors render correctly.

Install (once):

1. Open Blender → `Edit → Preferences → Add-ons`.
2. Click the dropdown arrow in the top-right corner of the Add-ons panel →
   `Install from Disk…`.
3. Select `import_ply_pointcloud_addon.py`.
4. Tick the checkbox next to **Import PLY as Colored Point Cloud** to enable
   it. It stays installed and enabled across Blender restarts.

Use (every time):

1. `File → Import → PLY as Colored Point Cloud (.ply)`.
2. Pick one or more `.ply` files (multi-select with Cmd/Ctrl or Shift).
3. Optional: set **Point Radius** in the left panel of the file dialog before
   confirming (default 0.005 m; smaller = finer dots, larger = solid-looking
   surfaces). To change it after import, re-import or edit the `radius`
   attribute.
4. Optional: **Round Points** (on by default) adds a shared Geometry Nodes
   modifier that instances smooth spheres on the points, so points look
   round in the viewport. Disable it to keep Blender's native point display.
5. Done. The add-on imports each file, converts it to a Point Cloud, assigns
   the shared `PointCloudRGB` material, and switches the viewport to Material
   Preview. Colors are visible immediately.

Why "Round Points"? Material Preview uses EEVEE, and EEVEE draws native
point clouds as very low-poly sphere instances — up close they look like
squares/diamonds ("pixels"). Only Cycles renders native points as true
spheres. The `PointCloudSpheres` modifier instances smooth-shaded ico
spheres instead, so points look round in every engine. If points still
look faceted at extreme close-up, raise **Sphere Subdivisions** on the
`Round Points` modifier (Properties → wrench icon), at the cost of
viewport performance on large clouds.

Notes:

- Needs Blender 4.2+; tested with Blender 5.2 LTS.
- If a file has no `red`/`green`/`blue` vertex properties, the add-on warns
  and the points render white.
- Colors show in Material Preview or Rendered shading only; the add-on
  switches Solid viewports to Material Preview automatically, and lowers
  the viewport Clip Start to 1 mm so you can zoom in close without
  geometry being clipped.
- The material reads `Col` through both the `Geometry` and the `Instancer`
  attribute paths and sums them, so it colors correctly with and without
  the Round Points modifier. Old `PointCloudRGB` materials from v1.0 are
  upgraded automatically on the next import.
- Point clouds imported with v1.0 don't get round points retroactively;
  the quickest fix is to delete and re-import them (or add the
  `PointCloudSpheres` node group as a Geometry Nodes modifier manually).
