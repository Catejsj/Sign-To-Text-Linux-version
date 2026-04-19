extends Node3D

var udp := PacketPeerUDP.new()
const PORT := 8888

var skeleton: Skeleton3D = null

const CACHE_FRAMES := 10
var bone_cache := {}

const BONE_MAP := {
	"upper_arm_l": "mixamorig_LeftArm",
	"upper_arm_r": "mixamorig_RightArm",
	"lower_arm_l": "mixamorig_LeftForeArm",
	"lower_arm_r": "mixamorig_RightForeArm",
	"hand_l":      "mixamorig_LeftHand",
	"hand_r":      "mixamorig_RightHand",
}

var packets_received := 0
var last_frame_id    := 0


func _ready() -> void:
	print("=".repeat(60))
	print("KHMER SIGN RECOGNIZER - GODOT VISUALIZATION")
	print("=".repeat(60))

	skeleton = _find_skeleton(self)
	if skeleton == null:
		push_error("❌ No Skeleton3D found!")
	else:
		print("✅ Skeleton found: ", skeleton.get_path())
		_print_bone_list()

	var err := udp.bind(PORT)
	if err == OK:
		print("✅ UDP listening on port ", PORT)
	else:
		push_error("❌ UDP bind failed (error %d)" % err)

	print("Waiting for bone data from WSL...")
	print("=".repeat(60))


func _process(_delta: float) -> void:
	while udp.get_available_packet_count() > 0:
		var raw  := udp.get_packet()
		var text := raw.get_string_from_utf8()
		var json := JSON.new()
		if json.parse(text) == OK:
			_apply_bone_data(json.get_data())
			packets_received += 1
			if packets_received % 60 == 0:
				print("📦 %d packets | frame %d" % [packets_received, last_frame_id])

	for key in bone_cache.keys():
		bone_cache[key]["ttl"] -= 1
		if bone_cache[key]["ttl"] <= 0:
			bone_cache.erase(key)


func _apply_bone_data(data: Dictionary) -> void:
	if data.has("frame_id"):
		last_frame_id = data["frame_id"]

	if skeleton == null:
		return

	# Apply main arm bones
	if data.has("bone_transforms"):
		var transforms: Dictionary = data["bone_transforms"]
		for wsl_key in BONE_MAP.keys():
			if not transforms.has(wsl_key):
				continue
			var rot_arr: Array = transforms[wsl_key]["rotation"]
			var q := Quaternion(rot_arr[0], rot_arr[1], rot_arr[2], rot_arr[3])
			q = q.normalized()
			var bone_name: String = BONE_MAP[wsl_key]
			_apply_rotation(bone_name, q)

	# Apply finger bones
	if data.has("finger_transforms"):
		var fingers: Dictionary = data["finger_transforms"]

		# Debug print every 60 packets
		if packets_received % 60 == 0:
			print("Finger count: ", fingers.size())
			if fingers.size() > 0:
				var first_key = fingers.keys()[0]
				print("Sample finger key: ", first_key)
				print("Sample finger data: ", fingers[first_key])

		for finger_key in fingers.keys():
			var finger_data: Dictionary = fingers[finger_key]
			var bone_name: String = finger_data["bone_name"]
			var rot_arr: Array = finger_data["rotation"]
			var q := Quaternion(rot_arr[0], rot_arr[1], rot_arr[2], rot_arr[3])
			q = q.normalized()
			_apply_rotation(bone_name, q)


func _apply_rotation(bone_name: String, q: Quaternion) -> void:
	var bone_idx := skeleton.find_bone(bone_name)

	if bone_idx == -1:
		if not bone_cache.has("warned_" + bone_name):
			push_warning("Bone not found: " + bone_name)
			bone_cache["warned_" + bone_name] = {"quat": Quaternion.IDENTITY, "ttl": 99999}
		return

	bone_cache[bone_name] = {"quat": q, "ttl": CACHE_FRAMES}

	var rest := skeleton.get_bone_rest(bone_idx)
	var new_pose := rest
	new_pose.basis = rest.basis * Basis(q)
	skeleton.set_bone_pose(bone_idx, new_pose)


func _find_skeleton(node: Node) -> Skeleton3D:
	if node is Skeleton3D:
		return node
	for child in node.get_children():
		var result := _find_skeleton(child)
		if result != null:
			return result
	return null


func _print_bone_list() -> void:
	print("── Skeleton bones ──")
	for i in range(skeleton.get_bone_count()):
		print("  [%d] %s" % [i, skeleton.get_bone_name(i)])
	print("────────────────────")
