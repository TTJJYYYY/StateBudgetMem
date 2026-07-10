from pathlib import Path
import json


ROOT = Path("data/memorybank_reproduction")

USER_DIR = ROOT / "users"

OUTPUT_DIR = Path(
    "results/memorybank/reproduction_storage"
)

OUTPUT_FILE = OUTPUT_DIR / "summary.json"


def build_summary_dataset():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result = []

    for user_file in sorted(USER_DIR.glob("*.json")):

        with open(
            user_file,
            "r",
            encoding="utf-8"
        ) as f:
            user = json.load(f)


        user_summary = {

            "user_id":
                user["user_id"],


            "summary_mode":
                "fixture",


            "portrait_mode":
                "fixture",


            "user_memory_ids": {

                "event_summary_id":
                    f"{user['user_id']}_global_event_summary",

                "portrait_id":
                    f"{user['user_id']}_global_user_portrait"
            },


            "days": []
        }

        for day in user["days"]:

            user_summary["days"].append(
                {

                    "date":
                        day["date"],


                    "daily_event_summary":
                        day.get(
                            "daily_event_summary",
                            ""
                        ),


                    "daily_personality":
                        day.get(
                            "daily_personality",
                            ""
                        ),


                    "source":
                        "manual_fixture"

                }
            )


        user_summary["global_event_summary"] = (
            user.get(
                "global_event_summary",
                ""
            )
        )


        user_summary["global_user_portrait"] = (
            user.get(
                "global_user_portrait",
                ""
            )
        )


        result.append(user_summary)


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        f"Saved fixture summary to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    build_summary_dataset()