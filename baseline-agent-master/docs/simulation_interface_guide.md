# 仿真接口说明表

仿真接口实现：`TongSimGrpcClient`  
默认地址：`127.0.0.1:50060`  
配置字段：`tongsim_server_endpoint`

## 角色生命周期

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `spawn_character` | `asset_name`, `loc`, `rot`, `desired_name`, `fov`, `width`, `height` | `character_id` | 在仿真中生成角色 |
| `destory_character` | `character_id` | `dict` | 销毁角色 |
| `heartbeat` | - | `dict` | 保持 TongSim 连接活跃 |
| `close` | - | - | 关闭 TongSim 连接并清理资源 |

## 感知查询

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `acquire_first_person_perception` | `character_id`, `width=None`, `height=None` | `dict`：`image`, `objects` | 一次获取第一视角组合图和映射后的可见物体信息 |
| `has_object_in_hand` | `character_id` | `(has_object, hand_idx)` | 查询是否持有物体以及被占用的手部索引；未持有时索引为 `None` |

`acquire_first_person_perception` 的返回值结构如下：

```python
{
    "image": "<base64 JPEG>",
    "objects": [
        {
            "object_id": "1",
            "color": "Green",
            "shape": "Cuboid",
            "place_location": {"X": 0.0, "Y": 0.0, "Z": 0.0},
            "world_aabb": {
                "min": {"X": 0.0, "Y": 0.0, "Z": 0.0},
                "max": {"X": 0.0, "Y": 0.0, "Z": 0.0},
            },
        }
    ],
}
```

- `image` 左侧为第一视角 RGB，右侧为带数字 ID 标注的语义分割图。
- `objects[*].object_id` 与右侧图中的数字标注一一对应，是后续物体动作唯一允许使用的映射 ID。客户端不会获得 TongSim SDK 的原始物体 ID，也不需要自行转换。
- `width` 和 `height` 指最终组合图的宽高，必须同时传入。省略时不缩放，组合图宽度为摄像机宽度的 2 倍，高度等于摄像机高度。按 `spawn_character` 的默认摄像机 720×1000 计算，组合图为 1440×1000；`VLMAgent` 默认使用 1280×720 摄像机，对应组合图为 2560×720。
- 旧的 `acquire_first_person_image`、`acquire_first_person_segmantic_image`、`fetch_first_person_visible_objects`、`get_object_basic_info`、`get_object_world_aabb`、`get_object_id_by_name` 和 `get_object_in_hand` 已移除。

## 视角控制

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `look_at_location` | `character_id`, `target_location`, `is_cancel`, `execute_immediately` | `dict` | 看向指定坐标 |
| `look_at_object` | `character_id`, `object_id`, `is_cancel` | `dict` | 看向指定物体 |
| `point_at_object` | `character_id`, `object_id`, `is_cancel`, `which_hand` | `dict` | 指向指定物体 |

## 移动与抓取

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `move_to_location` | `character_id`, `target_location`, `stop_distance` | `dict` | 移动到指定坐标 |
| `move_forward` | `character_id`, `distance` | `dict` | 向前移动指定距离 |
| `move_to_object` | `character_id`, `object_id` | `dict` | 移动到指定物体附近 |
| `move_and_take_object` | `character_id`, `object_id`, `which_hand` | `dict` | 移动到物体并抓取 |
| `turn_in_degree` | `character_id`, `degree` | `dict` | 原地旋转指定角度 |

## 放置与物体操作

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `put_down_to_location` | `character_id`, `target_location`, `which_hand`, `disable_physics`, `hold_if_unreachable`, `force_release`, `auto_rotate`, `rotation`, `force_locate` | `dict` | 将手中物体放到指定位置 |
| `move_and_put_down` | `character_id`, `move_target_location`, `put_target_location`, `which_hand`, `put_rotation` | `dict` | 移动到指定位置后放下手中物体 |
| `move_and_put_down_object_in_container` | `character_id`, `which_hand` | `dict` | 将手中物体放入容器 |
| `set_object_pose` | `object_id`, `location`, `rotation` | `bool` | 直接设置物体位置和旋转 |
| `pour_water` | `character_id`, `object_id`, `location`, `which_hand` | `dict` | 倒水 |
| `slice_food` | `character_id`, `object_id`, `location` | `dict` | 切食物 |
| `wash_hands` | `character_id`, `faucet_object_id` | `dict` | 洗手 |
| `wash_object_in_hand` | `character_id`, `faucet_object_id` | `dict` | 清洗手中物体 |

## 场景交互

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `open_door` | `character_id`, `door_id`, `which_hand` | `dict` | 开门 |
| `close_door` | `character_id`, `door_id`, `which_hand` | `dict` | 关门 |
| `sit_down_to_object` | `character_id`, `object_id` | `dict` | 坐到指定物体 |
| `mop_floor` | `character_id`, `dirt_id` | `dict` | 拖地 |
| `rest` | `character_id` | `dict` | 休息 |
| `speak_to_npc` | `character_id`, `target`, `content` | `dict` | 与 NPC 对话 |
| `interact` | `character_id`, `object_id`, `new_object_state` | `dict` | 切换物体状态 |

## 常用参数格式

| 参数 | 格式 | 示例 |
|---|---|---|
| `target_location` / `location` | 三维坐标列表或字典 | `[100.0, 200.0, 50.0]` |
| `rotation` / `put_rotation` | `roll`, `yaw`, `pitch` | `{"roll": 0, "yaw": 90, "pitch": 0}` |
| `which_hand` | 整数 | `0` |
| `object_id` | `acquire_first_person_perception` 返回的映射 ID | `"1"` |
