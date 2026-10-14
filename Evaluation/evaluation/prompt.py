import json


def make_prompt(
    bboxes,
    question,
    options,
    qa_type,
    dataset,
    use_video=False,
    use_bbox=True,
    use_video_2d=False,
    use_video_3d=False,
    use_bbox_2d=False,
    scene_description=None,
):
    if not use_video:
        use_bbox = True

    if dataset == "vsibench":
        if use_bbox and not use_video_3d and not use_video_2d and not use_bbox_2d:
            if not bboxes:
                print("Warning: Bboxes are empty for vsibench bbox mode")
            if scene_description:
                return _make_vsibench_desc_bbox_prompt(
                    bboxes, scene_description, question, options, qa_type
                )
            return _make_vsibench_bbox_prompt(bboxes, question, options, qa_type)
        elif use_bbox_2d:
            assert bboxes, "Bboxes are required for bbox_2d mode"
            return _make_vsibench_bbox_2d_prompt(bboxes, question, options, qa_type)
        elif use_video_3d:
            return _make_vsibench_video_bbox_prompt(bboxes, question, options, qa_type)
        elif use_video_2d:
            return _make_vsibench_video_2d_prompt(question, options, qa_type)
        elif use_video:
            return _make_vsibench_video_only_prompt(question, options, qa_type)

    elif dataset == "robospatial":
        if not use_video:
            return _make_robospatial_bbox_prompt(bboxes, question, options, qa_type)
        elif use_bbox:
            return _make_robospatial_video_bbox_prompt(
                bboxes, question, options, qa_type
            )
        else:
            return _make_robospatial_video_only_prompt(question, options, qa_type)


def _make_vsibench_bbox_2d_prompt(bboxes, question, options, qa_type):
    """
    Generate prompt for 2D bbox baseline experiment.
    bboxes: list of dicts with keys: box_2d [x1, y1, x2, y2], label, camera_pose (4x4 matrix)
    """
    # Format the 2D bounding boxes with camera poses
    bboxes_prompt = "The objects' 2D bounding boxes are:\n"
    for bbox in bboxes:
        bboxes_prompt += json.dumps(bbox) + "\n"

    bboxes_prompt += (
        "\nThe bounding boxes are given in this JSON format: "
        '`[{"box_2d": [x1, y1, x2, y2], "label": "category", "camera_pose": [[...4x4 matrix...]]}]`.\n\n'
        "**Data Explanation:**\n"
        "- `box_2d`: 2D bounding box in image coordinates (x1, y1) is top-left corner, (x2, y2) is bottom-right corner\n"
        "- `label`: Object category name\n"
        "- `camera_pose`: 4x4 camera extrinsic matrix (world-to-camera transformation)\n"
        "  - The matrix transforms world coordinates to camera coordinates\n"
        "  - Format: [[R11, R12, R13, tx], [R21, R22, R23, ty], [R31, R32, R33, tz], [0, 0, 0, 1]]\n"
        "  - R: 3x3 rotation matrix, t: 3x1 translation vector\n\n"
    )

    # Format the question
    question_prompt = ""
    if options and options != "N/A":
        question_prompt = (
            f"Based on the provided 2D bounding boxes and camera poses, "
            f"reason about the 3D spatial relationships and answer the following question:\n"
            f"{question}\nChoose from the options:\n{options}\n\n"
        )
    else:
        question_prompt = f"Question: {question}\n\n"

    # Add task-specific instructions based on question type
    if "direction" in qa_type:
        prompt_path = "./prompts_bbox_2d/direction.txt"
        try:
            with open(prompt_path, "r") as f:
                question_type_instruction = f.read()
        except FileNotFoundError:
            question_type_instruction = _get_default_2d_instruction(qa_type)
    elif "object_size" in qa_type:
        prompt_path = "./prompts_bbox_2d/object_size.txt"
        try:
            with open(prompt_path, "r") as f:
                question_type_instruction = f.read()
        except FileNotFoundError:
            question_type_instruction = _get_default_2d_instruction(qa_type)
    elif "rel_distance" in qa_type:
        prompt_path = "./prompts_bbox_2d/rel_distance.txt"
        try:
            with open(prompt_path, "r") as f:
                question_type_instruction = f.read()
        except FileNotFoundError:
            question_type_instruction = _get_default_2d_instruction(qa_type)
    elif "abs_distance" in qa_type:
        prompt_path = "./prompts_bbox_2d/abs_distance.txt"
        try:
            with open(prompt_path, "r") as f:
                question_type_instruction = f.read()
        except FileNotFoundError:
            question_type_instruction = _get_default_2d_instruction(qa_type)
    elif "counting" in qa_type:
        prompt_path = "./prompts_bbox_2d/counting.txt"
        try:
            with open(prompt_path, "r") as f:
                question_type_instruction = f.read()
        except FileNotFoundError:
            question_type_instruction = _get_default_2d_instruction(qa_type)
    elif "route" in qa_type:
        prompt_path = "./prompts_bbox_2d/route_planning.txt"
        try:
            with open(prompt_path, "r") as f:
                question_type_instruction = f.read()
        except FileNotFoundError:
            question_type_instruction = _get_default_2d_instruction(qa_type)
    else:
        question_type_instruction = _get_default_2d_instruction(qa_type)

    question_prompt += (
        "General Instructions:\n"
        "- Use the 2D bounding boxes and camera poses to infer 3D spatial relationships\n"
        "- The camera pose tells you the camera's position and orientation in the world\n"
        "- Larger 2D boxes typically mean objects are closer to the camera or physically larger\n"
        "- The position of boxes in the image relates to the viewing direction\n"
        "- Ensure your reasoning is concise and clear\n"
        "- When you have to do estimation, explain your reasoning based on the available data\n"
        "- Before you conclude, ensure your estimation aligns with all the instructions and guidelines provided\n"
        + question_type_instruction
    )

    return (
        "You are a multimodal reasoning model that interprets 2D bounding boxes and camera information. "
        "You will be provided with a list of 2D bounding boxes in image coordinates and their corresponding camera poses. "
        "Your goal is to infer 3D spatial relationships between objects, including positions, relative distances, "
        "directions, sizes, and other spatial properties based on the 2D observations and camera geometry.\n\n"
        + "\n\n"
        + bboxes_prompt
        + question_prompt
    )


def _get_default_2d_instruction(qa_type):
    """
    Get default instruction for 2D bbox reasoning when specific prompt file doesn't exist
    """
    return (
        "\n**Reasoning Strategy:**\n"
        "1. Identify relevant objects from the 2D bounding boxes\n"
        "2. Analyze the camera pose to understand the viewpoint\n"
        "3. Use 2D box sizes and positions to infer depth and 3D relationships\n"
        "4. Consider that:\n"
        "   - Objects higher in the image are often at similar depth or farther\n"
        "   - Larger boxes may indicate closer or larger objects\n"
        "   - Camera pose rotation affects how 3D positions project to 2D\n"
        "5. Provide your reasoning and final answer\n\n"
        "Output format:\n"
        "{{\n"
        "  'Reasoning': 'Based on the 2D bounding boxes and camera pose, ...',\n"
        "  'Answer': <answer>\n"
        "}}\n"
    )


def _make_vsibench_bbox_prompt(bboxes, question, options, qa_type):
    walls = [bbox for bbox in bboxes if bbox.get("label", "").lower() == "wall"]
    objects = [bbox for bbox in bboxes if bbox.get("label", "").lower() != "wall"]

    bboxes_prompt = ""
    # if walls and qa_type in ["route_planning"]:
    #     bboxes_prompt += "The walls are:\n"
    #     for wall in walls:
    #         bboxes_prompt += json.dumps(wall) + "\n"
    #     bboxes_prompt += "\n"
    bboxes_prompt += "The objects' bounding boxes are:\n"
    for bbox in objects:
        bboxes_prompt += json.dumps(bbox) + "\n"

    bboxes_prompt += '\nThe bounding boxes are given in world coordinate in this JSON format: `[{"bbox_3d": [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw], "label": "category"}]`.\n\n\n'

    question_prompt = ""
    if options and options != "N/A":
        question_prompt = f"Based on the provided bounding boxes, consider the spatial arrangement of all objects and answer the following question:\n {question} Choose from the options:\n {options}\n\n"
    else:
        question_prompt = f"Question: {question}\n\n"

    #  add typical failure cases
    if "direction" in qa_type or "object_size" in qa_type:
        bboxes_prompt += "You have to be careful with the bounding box data:\n"
        failure_cases_path = "./prompts/code/failure_cases.txt"
        with open(failure_cases_path, "r") as f:
            failure_cases = f.read()
        bboxes_prompt += failure_cases

    if "direction" in qa_type:
        prompt_path = "./prompts/code/direction.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "object_size" in qa_type:
        prompt_path = "./prompts/code/object_size.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "rel_distance" in qa_type:
        prompt_path = "./prompts/code/rel_distance.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "abs_distance" in qa_type:
        prompt_path = "./prompts/code/abs_distance.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "counting" in qa_type:
        prompt_path = "./prompts/code/counting.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "room" in qa_type:
        prompt_path = "./prompts/code/room_size.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "route" in qa_type:
        prompt_path = "./prompts/code/route_planning.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "order" in qa_type:
        question_type_instruction = (
            "Bounding boxes are given in appearance order (first appearance to last). "
            "Use the sequence of bounding boxes to determine the first-appearance order of the queried categories.\n"
            "Output format: \n {{'Reasoning': '...', 'Answer': <answer>}}."
        )
    else:
        question_type_instruction = "Output format: \n {{'Reasoning': 'According to the bounding box data, ... .', 'Answer': <answer>}}."

    question_prompt += (
        ("General Instructions:\n" if qa_type not in ["route_planning"] else "")
        + "- Think step-by-step and ensure your reasoning is concise and clear\n"
        + "- Before you conclude, ensure your estimation aligns with all the instructions and guidelines provided.\n"
        + (
            "- If you detect logical or numeric mistakes, revise your reasoning and provide the corrected answer.\n"
            if qa_type not in ["route_planning"]
            else ""
        )
        + (
            "- Ensure your answers are based on **EXPLICIT** math operations.\n"
            if qa_type not in ["route_planning"]
            else ""
        )
        + (
            "- You only have to focus on relevant objects and their relationships. Do not overthink.\n"
            if qa_type not in ["room_size_estimation", "route_planning"]
            else ""
        )
        + question_type_instruction
    )

    return (
        "You are a multimodal reasoning model that interprets structured scene inputs. "
        "You will be provided with a list of predicted bounding boxes representing objects in the scene. "
        "The bounding boxes are predicted by a 3D object detector and may not be perfect (e.g. some objects are missing, some objects are mislabeled). "
        "Your goal is to analyze the provided bounding boxes, recover the scene layout, and then reason about the spatial relationships. "
        + "\n\n"
        + bboxes_prompt
        + question_prompt
    )


def _make_vsibench_desc_bbox_prompt(
    bboxes, scene_description, question, options, qa_type
):
    walls = [bbox for bbox in bboxes if bbox.get("label", "").lower() == "wall"]
    objects = [bbox for bbox in bboxes if bbox.get("label", "").lower() != "wall"]

    bboxes_prompt = ""
    desc_prompt = ""
    # Handle dict-format graph data: extract the "objects" list which contains distance_meters
    scene_counting = None
    scene_appearance_order = None
    if isinstance(scene_description, dict) and "objects" in scene_description:
        scene_description_list = scene_description["objects"]
        if "counting" in scene_description:
            scene_counting = scene_description["counting"]
        if "appearance_order" in scene_description:
            scene_appearance_order = scene_description["appearance_order"]
        if "room_size" in scene_description:
            scene_room_size = scene_description["room_size"]
    elif isinstance(scene_description, list):
        scene_description_list = scene_description
    else:
        scene_description_list = scene_description

    desc_prompt = "## Scene Description\n"
    # Include room size if available
    if scene_room_size is not None:
        desc_prompt += (
            "### Room Size (in square meters)\nThe room size is estimated to be: "
            + str(scene_room_size)
            + " square meters\n\n"
        )
    # Include appearance order if available (used for obj_appearance_order task)
    if scene_appearance_order is not None:
        desc_prompt += "### Object Appearance Order (from visual observation)\nThe following list shows the order in which object categories first appear in the video, from first to last:\n"
        desc_prompt += json.dumps(scene_appearance_order) + "\n\n"
    # Include counting data if available
    if scene_counting is not None:
        desc_prompt += "### Object Counts (from visual observation)\nThe following counts represent the total number of distinct physical instances of each category observed in the scene:\n"
        desc_prompt += json.dumps(scene_counting, indent=2) + "\n\n"
    desc_prompt += "### Object Details\nThe following scene description contains object attributes, sizes (in centimeters), and the three nearest neighbors (ordered from nearest to farthest) for each object. Note: the actual total counts are given in the Object Counts section above.\n"
    if isinstance(scene_description_list, list):
        cleaned_list = []
        for obj in scene_description_list:
            obj_copy = {k: v for k, v in obj.items() if k != "relationships"}
            if "relationships" in obj and isinstance(obj["relationships"], list):
                sorted_rels = sorted(
                    obj["relationships"],
                    key=lambda r: r.get("distance_meters", float("inf")),
                )
                obj_copy["nearest_neighbors"] = [
                    r["neighbor"] for r in sorted_rels if "neighbor" in r
                ]
            cleaned_list.append(obj_copy)
        desc_prompt += json.dumps(cleaned_list, indent=2) + "\n\n"
    elif isinstance(scene_description_list, str):
        desc_prompt += scene_description_list + "\n\n"
    else:
        desc_prompt += json.dumps(scene_description_list, indent=2) + "\n\n"

    bboxes_prompt += "## Bounding Box Data\n"
    for bbox in objects:
        bboxes_prompt += json.dumps(bbox) + "\n"

    bboxes_prompt += '\nThe bounding boxes are given in world coordinate in this JSON format: `[{"bbox_3d": [x_center, y_center, z_center, x_full_size, y_full_size, z_full_size, roll, pitch, yaw], "label": "category"}]`.\n\n\n'
    bboxes_prompt += "You have to be careful with the bounding box data:\n"
    failure_cases_path = "./prompts/code/failure_cases.txt"
    with open(failure_cases_path, "r") as f:
        failure_cases = f.read()
    bboxes_prompt += failure_cases
    bboxes_prompt += "You should validate the bounding box data with the scene description to find the most reasonable bounding box data for each object."

    question_prompt = ""
    if options and options != "N/A":
        question_prompt = f"Based on the provided bounding boxes and scene description, consider the spatial arrangement of all objects and answer the following question:\n {question} Choose from the options:\n {options}\n\n"
    else:
        question_prompt = (
            f"You are required to output the numeric number. Question: {question}\n\n"
        )

    if "direction" in qa_type:
        prompt_path = "./prompts/code/direction.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "object_size" in qa_type:
        prompt_path = "./prompts/code/object_size.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "rel_distance" in qa_type:
        prompt_path = "./prompts/code/rel_distance.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "abs_distance" in qa_type:
        prompt_path = "./prompts/code/abs_distance.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "counting" in qa_type:
        prompt_path = "./prompts/code/counting.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "room" in qa_type:
        prompt_path = "./prompts/code/room_size.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "route" in qa_type:
        prompt_path = "./prompts/code/route_planning.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "order" in qa_type:
        question_type_instruction = (
            "The scene description contains an **'Object Appearance Order'** list that records the exact order "
            "in which categories first appear in the video. Use this list as your PRIMARY source to answer the question — "
            "find the ranks of the queried categories within that list and map them to the answer choices. "
            "Use the bounding boxes to validate the results."
            "Output format: \n {{'Reasoning': '...', 'Answer': <answer>}}."
        )
    else:
        question_type_instruction = (
            "Output format: \n {{'Reasoning': '...', 'Answer': <answer>}}."
        )

    question_prompt += (
        ("## General Instructions:\n" if qa_type not in ["route_planning"] else "")
        + "- Combine the scene description and the bounding box data to reason about the spatial relationships.\n"
        # + "- The scene description may contain `distance_meters` between object pairs — use these values for distance estimation and validation.\n"
        + "- Think step-by-step and ensure your reasoning is concise and clear\n"
        + "- Before you conclude, ensure your estimation aligns with all the instructions and guidelines provided.\n"
        + "- If some objects are not detected in bounding boxes, you MUST answer based on the scene description (especially `distance_meters` if available).\n"
        + "- When conflicts occur between the scene description and the bounding box data, you should answer based on the scene description.\n"
        + question_type_instruction
    )

    if qa_type in [
        "route_planning",
        "object_abs_distance",
        "object_rel_direction_hard",
        "object_rel_direction_easy",
        "object_rel_direction_medium",
    ]:
        return (
            "You are a multimodal reasoning model that interprets structured scene inputs. "
            "You will be provided with a list of predicted bounding boxes representing objects in the scene. "
            # "The scene description is provided to help you understand object relationships and choose appropriate objects for the question. You can always trust the scene description. "
            "The bounding boxes are predicted by a 3D object detector and may not be perfect (e.g. some objects are missing, some objects are mislabeled). "
            "Your goal is to analyze the provided information, recover the scene layout, and then reason about the spatial relationships. "
            + "\n\n"
            + desc_prompt
            + bboxes_prompt
            + question_prompt
        )
    else:
        return (
            "You are a multimodal reasoning model that interprets structured scene inputs. "
            "You will be provided with a scene description representing objects in the scene. "
            # "The scene description is provided to help you understand object relationships and choose appropriate objects for the question. You can always trust the scene description. "
            # "The bounding boxes are predicted by a 3D object detector and may not be perfect (e.g. some objects are missing, some objects are mislabeled). "
            "Your goal is to analyze the provided information, recover the scene layout, and then reason about the spatial relationships. "
            + "\n\n"
            + desc_prompt
            # + bboxes_prompt
            + question_prompt
        )


def _make_vsibench_video_2d_prompt(question, options, qa_type):
    question_prompt = ""
    if options and options != "N/A":
        question_prompt = f"Based on the provided video, consider the spatial arrangement of all objects and answer the following question:\n {question} Choose from the options:\n {options}\n\n"
    else:
        question_prompt = f"Question: {question}\n\n"

    if "direction" in qa_type:
        prompt_path = "./prompts_video/direction.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "object_size" in qa_type:
        prompt_path = "./prompts_video/object_size.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "rel_distance" in qa_type:
        prompt_path = "./prompts_video/rel_distance.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "abs_distance" in qa_type:
        prompt_path = "./prompts_video/abs_distance.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "counting" in qa_type:
        prompt_path = "./prompts_video/counting.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "route" in qa_type:
        prompt_path = "./prompts_video/route_planning.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    else:
        question_type_instruction = "Output format: \n {{'Reasoning': 'According to the video frames, ... .', 'Answer': <answer>}}."

    question_prompt += (
        "General Instructions:\n"
        "- You should first pay attention to the annotated bounding boxes. They are provided to help you locate and track the objects. Object categories are also labeled on the video frames."
        "- Ensure your reasoning is concise and clear\n"
        "- When you have to do numeric estimation, infer **explicit** numeric data from the video frames and relationships between objects. **Do not answer based on your own guess.**\n"
        "- Before you conclude, ensure your estimation aligns with all the instructions and guidelines provided.\n"
        "- If you detect logical or numeric mistakes, revise your reasoning and provide the corrected answer\n"
        + question_type_instruction
    )

    return (
        "You are a multimodal reasoning model that interprets structured scene inputs. "
        "You will be provided with a video showing objects in a scene from different viewpoints. "
        "Your goal is to find the objects, figure out their positions, sizes, orientations, and arrangement, and use the information to reason about spatial relationships, "
        "object appearance, object arrangement, and other spatial tasks."
        "To help locate and track the objects, bounding boxes are plotted on the video frames with the category name. \n\n"
        + question_prompt
    )


def _make_vsibench_video_bbox_prompt(bboxes, question, options, qa_type):
    walls = [bbox for bbox in bboxes if bbox.get("label", "").lower() == "wall"]
    objects = [bbox for bbox in bboxes if bbox.get("label", "").lower() != "wall"]

    bboxes_prompt = ""
    if walls and "room" in qa_type:
        bboxes_prompt = "The walls are:\n"
        for wall in walls:
            bboxes_prompt += json.dumps(wall) + "\n"
        bboxes_prompt += "\n"
    else:
        bboxes_prompt += "The objects' bounding boxes are:\n"
        for bbox in objects:
            bboxes_prompt += json.dumps(bbox) + "\n"

    bboxes_prompt += '\nThe bounding boxes are given in world coordinate in this JSON format: `[{"bbox_3d": [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw], "label": "category"}]`. Objects in the list are ordered by their appearance sequence in the video (first appearance to last appearance).\n\n\n'

    question_prompt = ""
    if options and options != "N/A":
        question_prompt = f"Based on the provided bounding boxes, consider the spatial arrangement of all objects and answer the following question:\n {question} Choose from the options:\n {options}\n\n"
    else:
        question_prompt = f"Question: {question}\n\n"

    if "direction" in qa_type:
        prompt_path = "./prompts/direction.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "object_size" in qa_type:
        prompt_path = "./prompts/object_size.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "rel_distance" in qa_type:
        prompt_path = "./prompts/rel_distance.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "abs_distance" in qa_type:
        prompt_path = "./prompts/abs_distance.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "counting" in qa_type:
        prompt_path = "./prompts/counting.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "room" in qa_type:
        prompt_path = "./prompts/room_size.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "route" in qa_type:
        prompt_path = "./prompts/route_planning.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    else:
        question_type_instruction = "Output format: \n {{'Reasoning': 'According to the video frames, ... . According to the bounding box data, ... . Comparing these two answers, my final answer is ...', 'Answer': <answer>}}."

    question_prompt += (
        "Please provide two INDEPENDENT answers in your reasoning: \n"
        "1. Answer solely based on video observations\n"
        "2. Answer solely based on bounding box data (**with explicit math operations**)\n\n"
        # "When the question is about a single object, you should prioritize the video observations. When the question is about multiple objects and their relationships, you should prioritize the bounding box data. \n\n"
        + "General Instructions:\n"
        "- Ensure your reasoning is concise and clear\n"
        "- Please provide two explicit answers in your reasoning, one based on video frames and one based on bounding box data. "
        "Do NOT let one method influence the other during inference—reason through each approach separately and completely before comparing. "
        "- When you have to do numeric estimation, you can trust the bounding box data or you can infer **explicit** numeric data from the video frames and relationships between objects. **Do not answer based on your own guess.**\n"
        "- Before you conclude, ensure your estimation aligns with all the instructions and guidelines provided.\n"
        "- If you detect logical or numeric mistakes, revise your reasoning and provide the corrected answer\n"
        + question_type_instruction
    )

    return (
        "You are a multimodal reasoning model that interprets structured scene inputs. "
        "You will be provided with a video showing objects in a scene from different viewpoints and a list of bounding boxes representing objects in the scene. "
        "Your goal is to find the objects, figure out their positions, sizes, orientations, and arrangement, and use the information to reason about spatial relationships, "
        "object appearance, object arrangement, and other spatial tasks.\n\n"
        + "\n\n"
        + bboxes_prompt
        + question_prompt
    )


def _make_vsibench_video_only_prompt(question, options, qa_type):
    question_prompt = ""
    if options and options != "N/A":
        question_prompt = f"Based on the provided video, consider the spatial arrangement of all objects and answer the following question:\n {question} Choose from the options:\n {options}\n\n"
    else:
        question_prompt = f"Question: {question}\n\n"

    if "direction" in qa_type:
        prompt_path = "./prompts/baseline/direction.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "object_size" in qa_type:
        prompt_path = "./prompts/baseline/object_size.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "rel_distance" in qa_type:
        prompt_path = "./prompts/baseline/rel_distance.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "abs_distance" in qa_type:
        prompt_path = "./prompts/baseline/abs_distance.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "counting" in qa_type:
        prompt_path = "./prompts/baseline/counting.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    elif "route" in qa_type:
        prompt_path = "./prompts/baseline/route_planning.txt"
        with open(prompt_path, "r") as f:
            question_type_instruction = f.read()
    else:
        question_type_instruction = "Output format: \n {{'Reasoning': 'According to the video frames, ... .', 'Answer': <answer>}}."

    question_prompt += (
        "General Instructions:\n"
        "- Ensure your reasoning is concise and clear\n"
        "- When you have to do numeric estimation, infer **explicit** numeric data from the video frames and relationships between objects. **Do not answer based on your own guess.**\n"
        "- Before you conclude, ensure your estimation aligns with all the instructions and guidelines provided.\n"
        # "- If you detect logical or numeric mistakes, revise your reasoning and provide the corrected answer\n"
        + question_type_instruction
    )

    return (
        "You are a multimodal reasoning model that interprets structured scene inputs. "
        "You will be provided with a video showing objects in a scene from different viewpoints. "
        "Your goal is to find the objects, figure out their positions, sizes, orientations, and arrangement, and use the information to reason about spatial relationships, "
        "object appearance, object arrangement, and other spatial tasks.\n\n"
        + "\n\n"
        + question_prompt
    )


def _make_robospatial_bbox_prompt(bboxes, question, options, qa_type):
    system_instruction = (
        """You are a multimodal reasoning model that analyzes 3D scenes from structured bounding box data.

INPUT FORMAT:
- You will receive a list of bounding boxes representing objects detected across multiple video frames
- Each bounding box contains: position, size, orientation, and predicted object category
- The same physical object may have multiple bounding boxes from different frames/viewpoints

KEY CHARACTERISTICS OF THE DATA:
1. Frame-based predictions: Objects appear across multiple frames with varying bounding boxes
   - Example: A table may have a small box (partial view of corner) in one frame and a large box (full view) in another frame
   
2. Position consistency: Bounding boxes of the same object should have similar positions in world coordinates, but may vary in:
   - Size (due to partial occlusion or viewing distance)
   - Orientation (due to different camera angles)

3. Category prediction noise: Object categories may be semantically similar but not exact
   - Example: "stool" might be labeled as "chair"
   - Strategy: If a queried category is missing, consider bounding boxes of semantically similar objects

YOUR TASK:
Use common sense reasoning to:
1. Based on the bounding box data observations, merge the bounding boxes of the same object from multiple frames to get the most accurate bounding box of the object
2. Infer object attributes (size, shape, material, etc.)
3. Determine spatial arrangements and relative positions between objects
4. Answer spatial reasoning questions about the scene

Apply multi-frame fusion and semantic understanding to provide accurate spatial analysis.\n\n
"""
        """**Coordinate System Conventions:**

**1. World Coordinate Frame:**
   - Z-axis: Points upward (vertical)
   - X-axis and Y-axis: Define the horizontal plane

**2. Rotation (Yaw) Convention:**
   - Reference direction: 0 radians aligns with the +X axis
   - Rotation direction: Positive yaw = counter-clockwise rotation
   - Rotation axis: All yaw rotations are around the Z-axis (vertical axis)

**3. Object-Centric Local Frame:**
   - Local +X axis → Object's FRONT direction
   - Local +Y axis → Object's LEFT direction
   - Note: These directions are relative to each object's own orientation, not the world frame

**Usage Guidelines:**
- When describing object orientations, use yaw angles relative to world +X axis
- When describing object-relative positions (e.g., "in front of the chair"), use the object's local frame where +X is front
"""
    )

    walls = [bbox for bbox in bboxes if bbox.get("label", "").lower() == "wall"]
    objects = [bbox for bbox in bboxes if bbox.get("label", "").lower() != "wall"]

    bboxes_str = ""

    bboxes_str += "The bounding boxes are:\n"
    for bbox in objects:
        bboxes_str += json.dumps(bbox) + "\n"

    bboxes_str += (
        "\n"
        "The bounding boxes are given in this JSON format: "
        'JSON: `[{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],"label":"category"}]`. '
        "Note: Objects in the list are ordered by their appearance sequence in the video (first appearance to last appearance).\n\n"
    )

    if walls and "room" in qa_type:
        bboxes_str += (
            "**Room & Wall Interpretation**\n"
            "- **Structure:** Bounding boxes with the same `room_id` belong to the same room. "
            "Within each room, `wall_index` provides the sequential order of walls from start to end.\n"
            "- **Data Imprecision:** Walls are thin segments and may be fragmented or have gaps. "
            "Use the `wall_index` sequence to infer the most plausible enclosed floor plan. "
            "**Small angle deviations (±5°) and gaps (<1m) are normal scan noise—do not reject calculations for these minor inconsistencies.**\n"
            "**If walls are too fragmented for precise tracing, estimate a 'mental model' from the wall positions and "
            "provide a reasonable area based on the overall X/Y spans, accounting for wall thickness.**\n"
            "\n"
        )
        bboxes_str += "The walls are:\n"
        for wall in walls:
            wall["bbox_3d"][4] = 0.25
            bboxes_str += json.dumps(wall) + "\n"
        bboxes_str += "\n"
        bboxes_str += "\n"

    question_str = f"Based on the provided bounding boxes, consider the spatial arrangement of all objects and answer the following question:\n {question}"

    if options and options != "N/A":
        question_str += f" Choose from the options:\n {options}\n\n"
    else:
        question_str += "\n\n"

    reasoning_instruction = (
        "**CRITICAL CONSTRAINTS:**\n"
        "- Your response MUST NOT exceed 1000 words (hard limit).\n"
        "- Despite the word limit, you MUST show ALL essential reasoning steps: extract coordinates, perform calculations, apply spatial logic, and verify your answer.\n"
        "- Be concise but complete—NO skipping critical steps to save words.\n\n"
        "- Double-check all reasoning steps and calculations before providing the final answer.\n"
        "IMPORTANT: You MUST end your response with exactly this format:\n"
        "Final Answer: Yes\n"
        "or\n"
        "Final Answer: No\n"
    )

    return system_instruction + bboxes_str + question_str + reasoning_instruction


def _make_robospatial_video_bbox_prompt(bboxes, question, options, qa_type):
    walls = [bbox for bbox in bboxes if bbox.get("label", "").lower() == "wall"]
    objects = [bbox for bbox in bboxes if bbox.get("label", "").lower() != "wall"]

    bboxes_prompt = ""
    if walls:
        bboxes_prompt = "The walls are:\n"
        for wall in walls:
            bboxes_prompt += json.dumps(wall) + "\n"
        bboxes_prompt += "\n"

    bboxes_prompt += "The objects' bounding boxes are:\n"
    for bbox in objects:
        bboxes_prompt += json.dumps(bbox) + "\n"

    bboxes_prompt += '\nThe bounding boxes are given in this JSON format: JSON: `[{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],"label":"category"}]`. Note: Objects in the list are ordered by their appearance sequence in the video (first appearance to last appearance).'

    question_prompt = ""
    if options and options != "N/A":
        question_prompt = f"Based on the provided bounding boxes, consider the spatial arrangement of all objects and answer the following question:\n {question} Choose from the options:\n {options}\n\n"
    else:
        question_prompt = f"Question: {question}\n\n"

    question_prompt += (
        "The reasoning steps should be clear and concise. "
        "Please double-check your calculations and spatial reasoning before providing the final answer. "
        "If you find any mistakes in your reasoning, correct them and provide the accurate answer. "
        "Format your final answer as: {{'Reasoning': '...', 'Answer': '<answer>'}}.\n\n"
    )

    return (
        "You are a multimodal reasoning model that interprets structured scene inputs. "
        "You will be provided with a video showing objects in a scene from different viewpoints and a list of bounding boxes representing objects in the scene. "
        "Your goal is to understand the spatial arrangement and use it to reason about spatial relationships, "
        "object interactions, depth estimation, and other spatial tasks.\n\n"
        "**Coordinate System Conventions:**\n"
        "1. World Frame: Z-axis points Up. X and Y are horizontal.\n"
        "2. Yaw: 0 rad is along +X. Positive Yaw is counter-clockwise. All rotations are around the Z-axis.\n"
        "**Reasoning Requirements:**\n"
        "Perspective: 'The second object' refers to the second object mentioned in the question text. "
        "Use its position as the origin and its Yaw to define local Forward (+X) and Left (+Y) directions.\n\n"
        "- To judge 'Front/Behind/Left/Right', you MUST perform a coordinate transformation from world to the reference object's local frame.\n"
        "- To judge 'Above/Below', compare the Z-center coordinates directly.\n"
        "- Do not assume objects must be physically touching or aligned to have a spatial relationship.\n\n"
        "- Fit Logic: 'Can Object A fit [direction] of Object B' asks about spatial possibility. "
        "It means: Is there enough unoccupied space in that specific direction relative to Object B to place Object A's bounding box without intersecting with any other existing objects in the scene?\n"
        "- When judging if A can fit beside B, do not forget to check the influence of ALL other objects in the scene.\n"
        + bboxes_prompt
        + question_prompt
    )


def _make_robospatial_video_only_prompt(question, options, qa_type):
    question_prompt = ""
    if options and options != "N/A":
        question_prompt = f"Based on the provided video, consider the spatial arrangement of all objects and answer the following question:\n {question} Choose from the options:\n {options}\n\n"
    else:
        question_prompt = f"Question: {question}\n\n"

    question_prompt += (
        "The reasoning steps should be clear and concise. "
        "Please double-check your calculations and spatial reasoning before providing the final answer. "
        "If you find any mistakes in your reasoning, correct them and provide the accurate answer. "
        "Format your final answer as: {{'Reasoning': '...', 'Answer': '<answer>'}}.\n\n"
    )

    return (
        "You are a multimodal reasoning model that interprets structured scene inputs. "
        "You will be provided with a video showing objects in a scene from different viewpoints. "
        "Your goal is to understand the spatial arrangement and use it to reason about spatial relationships, "
        "object interactions, depth estimation, and other spatial tasks.\n\n"
        + question_prompt
    )
