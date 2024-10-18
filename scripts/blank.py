import argparse
import json
import os


def make_blank(data):
    for dialogue in data:
        for turn in dialogue["turns"]:
            if turn["speaker"] != "USER":
                continue
            for frame in turn["frames"]:
                del frame["actions"]
                frame["slots"] = []
                frame["state"] = {
                    "active_intent": "",
                    "requested_slots": [],
                    "slot_values": {},
                }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input directory", required=True)
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    args = parser.parse_args()
    for root, dirs, files in os.walk(args.input):
        for file in files:
            if file.startswith("dialogues"):
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    make_blank(data)
                with open(os.path.join(args.output, file), "w") as f:
                    json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
