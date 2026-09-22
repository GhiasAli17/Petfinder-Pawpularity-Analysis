# # src/user_evaluation.py


import os
import numpy as np
import pandas as pd


def build_user_eval_images(selected_cases_df):
    user_eval_images_df = selected_cases_df.copy()

    user_eval_images_df = (
        user_eval_images_df[
            [
                "Id",
                "fold",
                "case_group",
                "case_type",
                "score_range",
                "true_paw",
                "pred_paw",
                "absolute_error",
            ]
        ]
        .sort_values(["case_group", "case_type", "Id"])
        .reset_index(drop=True)
    )

    return user_eval_images_df


def build_condition_materials(user_eval_images_df, vlm_feedback_df, img_folder):
    user_eval_materials_df = user_eval_images_df.merge(
        vlm_feedback_df[["Id", "vlm_model", "vlm_feedback"]],
        on="Id",
        how="left",
    )

    if user_eval_materials_df["vlm_feedback"].isna().any():
        missing_ids = user_eval_materials_df[
            user_eval_materials_df["vlm_feedback"].isna()
        ]["Id"].tolist()
        raise ValueError(f"Missing VLM feedback for IDs: {missing_ids}")

    condition_rows = []

    for row in user_eval_materials_df.itertuples(index=False):
        image_path = os.path.join(img_folder, row.Id + ".jpg")

        condition_rows.extend([
            {
                "Id": row.Id,
                "condition": "A",
                "condition_name": "Score only",
                "image_path": image_path,
                "pred_paw": row.pred_paw,
                "show_aux_feedback": False,
                "show_vlm_feedback": False,
                "show_gradcam": False,
                "vlm_model": "",
                "vlm_feedback": "",
            },
            {
                "Id": row.Id,
                "condition": "B",
                "condition_name": "Score + auxiliary feedback",
                "image_path": image_path,
                "pred_paw": row.pred_paw,
                "show_aux_feedback": True,
                "show_vlm_feedback": False,
                "show_gradcam": False,
                "vlm_model": "",
                "vlm_feedback": "",
            },
            {
                "Id": row.Id,
                "condition": "C",
                "condition_name": "Score + VLM feedback",
                "image_path": image_path,
                "pred_paw": row.pred_paw,
                "show_aux_feedback": False,
                "show_vlm_feedback": True,
                "show_gradcam": False,
                "vlm_model": row.vlm_model,
                "vlm_feedback": row.vlm_feedback,
            },
            {
                "Id": row.Id,
                "condition": "D",
                "condition_name": "Score + auxiliary feedback + Grad-CAM",
                "image_path": image_path,
                "pred_paw": row.pred_paw,
                "show_aux_feedback": True,
                "show_vlm_feedback": False,
                "show_gradcam": True,
                "vlm_model": "",
                "vlm_feedback": "",
            },
        ])

    return pd.DataFrame(condition_rows)


def build_counterbalanced_versions(
    user_eval_images_df,
    condition_materials_df,
    condition_order=("A", "B", "C", "D"),
):
    condition_lookup = condition_materials_df.set_index(["Id", "condition"])

    image_order_df = (
        user_eval_images_df
        .copy()
        .sort_values(["case_group", "case_type", "Id"])
        .reset_index(drop=True)
    )

    image_order_df["image_order"] = np.arange(len(image_order_df))

    participant_version_rows = []

    for version_index in range(len(condition_order)):
        version_name = f"Version {version_index + 1}"

        for row in image_order_df.itertuples(index=False):
            assigned_condition = condition_order[
                (int(row.image_order) + version_index) % len(condition_order)
            ]

            material = condition_lookup.loc[(row.Id, assigned_condition)]

            participant_version_rows.append({
                "version": version_name,
                "version_index": version_index + 1,
                "trial_order": int(row.image_order) + 1,
                "Id": row.Id,
                "fold": row.fold,
                "case_group": row.case_group,
                "case_type": row.case_type,
                "score_range": row.score_range,
                "true_paw": row.true_paw,
                "pred_paw": row.pred_paw,
                "absolute_error": row.absolute_error,
                "condition": assigned_condition,
                "condition_name": material["condition_name"],
                "image_path": material["image_path"],
                "show_aux_feedback": material["show_aux_feedback"],
                "show_vlm_feedback": material["show_vlm_feedback"],
                "show_gradcam": material["show_gradcam"],
                "vlm_model": material["vlm_model"],
                "vlm_feedback": material["vlm_feedback"],
            })

    return pd.DataFrame(participant_version_rows)


def build_questionnaire_items():
    likert_scale = "1 = strongly disagree, 7 = strongly agree"

    questionnaire_items = [
        {
            "item_id": "Q1_understandable",
            "question": "The system output was easy to understand.",
            "scale": likert_scale,
        },
        {
            "item_id": "Q2_useful",
            "question": "The system output was useful for interpreting the pet photo.",
            "scale": likert_scale,
        },
        {
            "item_id": "Q3_relevant",
            "question": "The feedback was relevant to visible characteristics of the photo.",
            "scale": likert_scale,
        },
        {
            "item_id": "Q4_actionable",
            "question": "The feedback gave useful guidance for improving the photo.",
            "scale": likert_scale,
        },
        {
            "item_id": "Q5_trust",
            "question": "The output helped me judge whether the prediction should be trusted.",
            "scale": likert_scale,
        },
        {
            "item_id": "Q6_too_much_information",
            "question": "The output contained too much information.",
            "scale": likert_scale,
        },
        {
            "item_id": "Q7_future_use",
            "question": "I would want to use this type of feedback in a pet-photo review system.",
            "scale": likert_scale,
        },
        {
            "item_id": "Q8_open_comment",
            "question": "Optional comment: What was helpful, unclear, or missing?",
            "scale": "Free text",
        },
    ]

    return pd.DataFrame(questionnaire_items)


def build_survey_export(user_eval_versions_df, questionnaire_df):
    survey_export_df = user_eval_versions_df.copy()

    for item in questionnaire_df.itertuples(index=False):
        survey_export_df[item.item_id] = ""

    def build_condition_display_text(row):
        lines = [f"Predicted Pawpularity score: {float(row.pred_paw):.1f}"]

        if bool(row.show_aux_feedback):
            lines.append("[Show task-specific auxiliary feedback generated by the prototype]")

        if bool(row.show_vlm_feedback):
            lines.append("[Show VLM-generated feedback]")

        if bool(row.show_gradcam):
            lines.append("[Show Grad-CAM attention map]")

        return "\n".join(lines)

    survey_export_df["display_instruction"] = survey_export_df.apply(
        build_condition_display_text,
        axis=1,
    )

    base_cols = [
        "version",
        "version_index",
        "trial_order",
        "Id",
        "image_path",
        "condition",
        "condition_name",
        "pred_paw",
        "show_aux_feedback",
        "show_vlm_feedback",
        "show_gradcam",
        "vlm_model",
        "vlm_feedback",
        "display_instruction",
    ]

    question_cols = questionnaire_df["item_id"].tolist()

    return survey_export_df[base_cols + question_cols].copy()


def build_design_summary(
    user_eval_images_df,
    user_eval_versions_df,
    condition_order,
    vlm_model,
):
    design_summary_rows = [
        {
            "item": "Number of selected images",
            "value": user_eval_images_df["Id"].nunique(),
        },
        {
            "item": "Number of participant versions",
            "value": user_eval_versions_df["version"].nunique(),
        },
        {
            "item": "Trials per participant version",
            "value": user_eval_versions_df.groupby("version")["Id"].nunique().iloc[0],
        },
        {
            "item": "Experimental conditions",
            "value": ", ".join(condition_order),
        },
        {
            "item": "VLM baseline model",
            "value": vlm_model,
        },
        {
            "item": "VLM prompt type",
            "value": (
                "One fixed prompt template with image-specific displayed "
                "auxiliary attributes"
            ),
        },
        {
            "item": "Evaluation scale",
            "value": "7-point Likert scale plus optional free-text comment",
        },
    ]

    return pd.DataFrame(design_summary_rows)


def get_design_note():
    return """
User-evaluation design note:

This implementation uses held-out example images as the user-evaluation set.
The four experimental conditions are counterbalanced across four participant
versions. If the number of images is not divisible by 4, the number of trials
per condition will be slightly uneven within each participant version.

For the final user study, 16 or 20 selected images are recommended to obtain
equal condition counts.
""".strip()


def get_methods_text():
    return """
VLM comparison baseline and pilot user-evaluation setup

To compare the proposed task-specific selective-feedback pipeline with a
general-purpose vision-language model baseline, we added a minimal VLM
comparison at the inference-feedback stage. The VLM was not trained and was
not used for Pawpularity prediction. The Pawpularity score still comes from
the trained Pawpularity model.

For each selected held-out image, the VLM was asked to generate feedback only
for the visual attributes displayed by the selective auxiliary-feedback policy.
This keeps the VLM baseline aligned with the task-specific auxiliary feedback.
The VLM was given the image and the attribute names, but not the auxiliary
model's positive or negative decisions.

Four planned interface conditions are represented in the exported materials:

A. Score only
B. Score plus task-specific selective auxiliary feedback
C. Score plus VLM-generated feedback
D. Score plus task-specific selective auxiliary feedback and Grad-CAM

These exported materials are intended for the later live user-evaluation
interface. No participant responses are collected by this file.
""".strip()


def save_user_eval_outputs(
    active_dir,
    vlm_feedback_df=None,
    condition_materials_df=None,
    user_eval_versions_df=None,
    questionnaire_df=None,
    survey_export_df=None,
    design_summary_df=None,
    design_note=None,
    methods_text=None,
):
    paths = {}

    if vlm_feedback_df is not None:
        path = os.path.join(active_dir, "vlm_feedback_completed.csv")
        vlm_feedback_df.to_csv(path, index=False)
        paths["vlm_feedback"] = path

    if condition_materials_df is not None:
        path = os.path.join(active_dir, "user_eval_condition_materials.csv")
        condition_materials_df.to_csv(path, index=False)
        paths["condition_materials"] = path

    if user_eval_versions_df is not None:
        path = os.path.join(active_dir, "user_eval_counterbalanced_versions.csv")
        user_eval_versions_df.to_csv(path, index=False)
        paths["counterbalanced_versions"] = path

    if questionnaire_df is not None:
        path = os.path.join(active_dir, "user_eval_questionnaire_items.csv")
        questionnaire_df.to_csv(path, index=False)
        paths["questionnaire"] = path

    if survey_export_df is not None:
        path = os.path.join(active_dir, "user_eval_survey_export.csv")
        survey_export_df.to_csv(path, index=False)
        paths["survey_export"] = path

    if design_summary_df is not None:
        path = os.path.join(active_dir, "user_eval_design_summary.csv")
        design_summary_df.to_csv(path, index=False)
        paths["design_summary"] = path

    if design_note is not None:
        path = os.path.join(active_dir, "user_eval_design_note.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(design_note)
        paths["design_note"] = path

    if methods_text is not None:
        path = os.path.join(active_dir, "vlm_user_eval_methods_text.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(methods_text)
        paths["methods_text"] = path

    return paths


def check_user_eval_outputs(active_dir):
    expected_output_files = {
        "VLM feedback": os.path.join(active_dir, "vlm_feedback_completed.csv"),
        "Condition materials": os.path.join(active_dir, "user_eval_condition_materials.csv"),
        "Counterbalanced versions": os.path.join(active_dir, "user_eval_counterbalanced_versions.csv"),
        "Questionnaire items": os.path.join(active_dir, "user_eval_questionnaire_items.csv"),
        "Survey export": os.path.join(active_dir, "user_eval_survey_export.csv"),
        "Design summary": os.path.join(active_dir, "user_eval_design_summary.csv"),
        "Design note": os.path.join(active_dir, "user_eval_design_note.txt"),
        "Methods text": os.path.join(active_dir, "vlm_user_eval_methods_text.txt"),
    }

    sanity_rows = []

    for name, path in expected_output_files.items():
        sanity_rows.append({
            "output": name,
            "path": path,
            "exists": os.path.exists(path),
        })

    return pd.DataFrame(sanity_rows)