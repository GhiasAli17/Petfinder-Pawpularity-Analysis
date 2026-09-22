# src/vlm_feedback.py

import os
import pandas as pd
from PIL import Image
from google import genai


AUX_TASK_TO_VLM_LABEL = {
    "Eyes": "Eyes visibility",
    "Face": "Face visibility",
    "Occlusion": "Occlusion",
    "Blur": "Blur",
    "Subject Focus": "Subject focus",
    "Action": "Action/motion cues",
    "Accessory": "Accessory cues",
    "Near": "Near-camera view",
    "Group": "Multiple-pet/group cues",
    "Collage": "Collage/layout cues",
    "Human": "Visible-human cues",
    "Info": "Informational visual cues",
}

def generate_vlm_feedback_gemini(client, image_path, prompt, model):
    image = Image.open(image_path)

    response = client.models.generate_content(
        model=model,
        contents=[prompt, image],
    )

    return response.text.strip()


def generate_vlm_feedback_table(
    vlm_template_df,
    api_key,
    model="models/gemini-flash-lite-latest",
):
    client = genai.Client(api_key=api_key)
    vlm_rows = []

    for row in vlm_template_df.itertuples(index=False):
        print("Generating VLM feedback for:", row.Id)

        feedback = generate_vlm_feedback_gemini(
            client=client,
            image_path=row.image_path,
            prompt=row.vlm_prompt,
            model=model,
        )

        vlm_rows.append({
            "Id": row.Id,
            "fold": row.fold,
            "case_group": row.case_group,
            "case_type": row.case_type,
            "image_path": row.image_path,
            "displayed_aux_tasks": row.displayed_aux_tasks,
            "n_displayed_aux_tasks": row.n_displayed_aux_tasks,
            "vlm_model": model,
            "vlm_prompt": row.vlm_prompt,
            "vlm_feedback": feedback,
        })

    return pd.DataFrame(vlm_rows)


def save_vlm_feedback(vlm_feedback_df, active_dir, filename="vlm_feedback_completed.csv"):
    path = os.path.join(active_dir, filename)
    vlm_feedback_df.to_csv(path, index=False)
    return path

def get_displayed_aux_tasks_for_image(img_id, batch_results_df, aux_tasks):
    row = batch_results_df[batch_results_df["Id"] == img_id].iloc[0]
    displayed_tasks = []

    for task in aux_tasks:
        prefix = task.lower()
        if row[f"{prefix}_prototype_state"] != "abstain":
            displayed_tasks.append(task)

    return displayed_tasks


def get_vlm_feedback_prompt(displayed_tasks, include_photo_suggestion=True):
    attribute_lines = [
        f"- {AUX_TASK_TO_VLM_LABEL[task]}"
        for task in displayed_tasks
    ]

    if len(attribute_lines) == 0:
        attribute_text = (
            "- No task-specific attributes were displayed by the "
            "selective auxiliary-feedback policy."
        )
    else:
        attribute_text = "\n".join(attribute_lines)

    suggestion_instruction = (
        "\nAlso provide one short photo-review suggestion."
        if include_photo_suggestion
        else ""
    )

    return f"""
You are evaluating a pet photo for a user-facing image review interface.

Analyze only visible image characteristics. Do not mention breed, cuteness,
adoption likelihood, or predicted popularity.

The task-specific auxiliary-feedback system displayed feedback for the
following attributes for this image:

{attribute_text}

For each listed attribute, give one concise judgment using cautious language,
such as "appears", "may", or "seems".{suggestion_instruction}

Return only the requested lines.
Do not claim that changing the image will improve the Pawpularity score.
Keep the full response concise.
""".strip()


def build_vlm_template(
    user_eval_images_df,
    img_folder,
    batch_results_df,
    aux_tasks,
    include_photo_suggestion=True,
):
    rows = []

    for row in user_eval_images_df.itertuples(index=False):
        displayed_tasks = get_displayed_aux_tasks_for_image(
            img_id=row.Id,
            batch_results_df=batch_results_df,
            aux_tasks=aux_tasks,
        )

        vlm_prompt = get_vlm_feedback_prompt(
            displayed_tasks=displayed_tasks,
            include_photo_suggestion=include_photo_suggestion,
        )

        rows.append({
            "Id": row.Id,
            "fold": row.fold,
            "case_group": row.case_group,
            "case_type": row.case_type,
            "score_range": row.score_range,
            "true_paw": row.true_paw,
            "pred_paw": row.pred_paw,
            "absolute_error": row.absolute_error,
            "image_path": os.path.join(img_folder, row.Id + ".jpg"),
            "displayed_aux_tasks": ", ".join(displayed_tasks),
            "n_displayed_aux_tasks": len(displayed_tasks),
            "vlm_prompt": vlm_prompt,
        })

    return pd.DataFrame(rows)