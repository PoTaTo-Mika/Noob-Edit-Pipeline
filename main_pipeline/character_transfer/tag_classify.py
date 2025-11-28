# 将tag串分组为openpose需要的输入形式（？
import json

def process_tags_list(tags_list: str) -> list:
    if not tags_list:
        return []
    
    tags = [tag.strip() for tag in tags_list.split(' ')]
    tags = [tag for tag in tags if tag]

    return tags

def tag_classify(tags_list, json_list):
    # 处理标签列表
    processed_tags = process_tags_list(tags_list)
    # 初始化分类结果字典，所有类别初始值为"None"
    classified_result = {}
    all_categories = set(json_list.values())
    for category in all_categories:
        classified_result[category] = "None"
    # 添加unknown类别
    classified_result["unknown"] = "None"
    # 分类标签
    unknown_tags = []
    for tag in processed_tags:
        if tag in json_list:
            category = json_list[tag]
            if classified_result[category] == "None":
                classified_result[category] = tag
            else:
                # 如果该类别已有标签，用逗号分隔添加新标签
                classified_result[category] += f",{tag}"
        else:
            unknown_tags.append(tag)
    # 处理unknown标签
    if unknown_tags:
        classified_result["unknown"] = ",".join(unknown_tags)
    
    return classified_result

if __name__ == "__main__":
    # set up json to memorys
    with open('checkpoints/tags/style_transfer_shuffle.json') as f:
        TAG_CATEGORIES = json.load(f)
    test_tag = "1boy 1girl absurdres anklet areola_slip armpits arms_up ass assertive_female azatychan bar_censor bare_shoulders blush bombergirl bracelet breasts bridal_gauntlets censored chain circlet clothed_sex commentary_request covered_nipples cowgirl_position demon_girl demon_horns demon_tail demon_wings detached_sleeves girl_on_top grim_aloe halterneck heart heart_tattoo hetero highres horns jewelry kneehighs leggings long_hair looking_at_viewer looking_back mouth_veil multicolored_horns open_mouth outstretched_arms pelvic_curtain penis pink_leggings pussy quiz_magic_academy quiz_magic_academy_the_world_evolve red_eyes red_horns red_tail red_wings ring sex shoulder_tattoo sidelocks small_breasts smile socks"
    print(process_tags_list(test_tag))
    print(tag_classify(test_tag,TAG_CATEGORIES))