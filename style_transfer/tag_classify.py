# 将tag串分组为openpose需要的输入形式（？
import json

def process_tags_list(tags_list: str) -> list:
    if not tags_list:
        return []
    
    tags = [tag.strip() for tag in tags_list.split(',')]
    tags = [tag for tag in tags if tag]

    return tags

def tag_classify(tag,json_list):
    pass

if __name__ == "__main__":
    # set up json to memorys
    with open('checkpoints/tags/style_transfer_shuffle.json') as f:
        TAG_CATEGORIES = json.load(f)
    
