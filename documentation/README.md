



## Transformation convention

In the code we use the notation: `H_points_stereo_rl_from_polarized_left`. This is the homogeneous transformation of a point in the reference frame of the left polarized camera optical frame to become a point described in the left camera on the right stereo camera. In mathematical terms it is:

$p^{srl} = H^{srl}_{pl} p^{pl}$

where 
* $p^{srl}$ is a point described relative to the frame of the left camera in the right stereo camera on the dual stereo camera setup. 
* $H^{srl}_{pl}$ is the homogeneous transform. 
* $p^{pl}$ is a point described relative to the frame of the left polarized stereo camera. 

Coordinate frames:
* The coordinate frames of the cameras have x-axis to the right, y-axis down and z-axis forward. 
* The coordinate frame of the LiDAR has the z-axis point upward. 
* The `piren` frame is a north-east-down frame. 
* The `vessel` frame as x-axis forward, y-axis to the port side and the z-axis upward. However, note that we placed the sensors on the aft of the ferry prototype. This detail should not matter for the use of the dataset. 

Some of these can also be seen in the animations on the [main README.md](../README.md) where the x-axis is red, the y-axis is green and the z-axis is blue. 




## Table for each file type
Here is a table for explaining the columns and content of the different file types:

| File type | Location | Fields | Units | Coordinate frame | Example | Timestamp / join key |
|-----------|----------|--------|-------|------------------|---------|----------------------|
| `.pos` | `gnss/**/` | `date`, `time`, `latitude(deg)`, `longitude(deg)`, `height(m)`, `Q`, `ns`, `sdn(m)`, `sde(m)`, `sdu(m)`, `sdne(m)`, `sdeu(m)`, `sdun(m)`, `age(s)`, `ratio` | as named in columns | Geodetic WGS84 input; converted by code to local PIREN NED | `2024/07/17 10:21:37.200 63.4342 10.3921 34.19 1 15 ...` | code parses `date+time` to Unix ns, applies leap-second offset, then nearest-timestamp join to other sensors |
| `.jpg` | `images/**/` | filename-encoded `timestamp_ns` and side (`_left` / `_right`), JPEG payload | ns, side enum, 8-bit/ch | camera optical frames (left/right) | `1721211646279279000_left.jpg` | the timestamps are on trigger time, Unix ns;  `StereoCamera` uses exact same `timestamp_ns` for left-right pairing; cross-sensor alignment is nearest timestamp |
| `.csv` (camera calibration) | `images/**/`, `polarized/parameters/` | `K_*.csv`/`K_*.csv`: 3x3 intrinsics; `D_*.csv`/`dist_*.csv`: distortion coeffs; `R.csv`: 3x3 rotation; `t.csv`/`T.csv`: translation; optional `E.csv`, `F.csv` | pixels for intrinsics principal/focal terms, meters for translation, otherwise unitless | right -> left stereo extrinsics for `R` + `t/T` | `K_left.csv`: `731.89,0,939.15 / 0,731.89,511.22 / 0,0,1` | static metadata in same folder as image stream |
| `.csv` (`ins.csv`) | `lidar_and_ins/**/ins.csv` | tab-separated `timestamp_ns, n, e, d, qx, qy, qz, qw` | ns, m, unit quaternion | PIREN NED | `1721208033499757568\t1.23\t-0.45\t0.67\t0.0\t0.0\t0.0\t1.0` | timestamp is direct key for nearest join with LiDAR/cameras/GNSS; timestamp is always Unix ns and here it is also in trigger time  |
| `.bytes` | `lidar_and_ins/**/` | packed PointCloud2-like records, `point_step=48`, fields at byte offsets: `x(0,float32)`, `y(4,float32)`, `z(8,float32)`, `intensity(16,float32)`, `t(20,uint32)`, `reflectivity(24,uint16)`, `ring(26,uint16)`, `ambient(28,uint16)`, `range(32,uint32)` | m for xyz; other fields are sensor-native numeric channels | `lidar_aft/os_sensor` (from embedded metadata) | `1721207717241221888.bytes` | file stem is scan `timestamp_ns`; code currently returns `t,x,y,z` per point for downstream timing compensation; `t` is trigger time addition to the first trigger time that is in the file stem in Unix ns |
| `.json` | `polarized/*_images/*.json` | top-level array with exactly two camera records; each record contains `frame_id`, `timestamp_ns`, `cam_serial`, `Gain`, `ExposureTime`, `balance_ratio.{Red,Green,Blue}`, `incomplete` | ns for `timestamp_ns`, us for `ExposureTime`, others sensor-native | polarized left/right cameras | `[{"frame_id":1,"timestamp_ns":...,"cam_serial":"220300007",...},{...}]` | key used by code is `timestamp_mean = int((timestamp_ns[0] + timestamp_ns[1]) / 2)`; this maps to `.raw` by filename stem; each of these timestamps were the trigger Unix ns time for each of the polarized cameras in the polarized sensor rig |
| `.raw` | `polarized/*_images/*.raw` | packed 12-bit Bayer payload, decoded in code to a `(4096, 2448)` array before splitting/processing | 12-bit digital levels | polarized rig raw sensor grid (later rectified to left/right outputs) | `000001.raw` | joined to `.json` by identical filename stem; timestamp comes indirectly from paired `.json` |
