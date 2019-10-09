from gym_unity.envs import ActionFlattener

action_names = [[" ", "forward", "backward"], [" ", "right", "left"], [" ", "rot_left", "rot_right"],
                ["dont charge", "charge"]]
# OK! These seem right.

flattened_names = ActionFlattener([3, 3, 3, 2])

def convert_action_id_to_name(id):
    un_flattened_indices = flattened_names.lookup_action(id)
    name = ""
    action_counter=0
    for i in un_flattened_indices:
        name+=action_names[action_counter][i]
        action_counter+=1
        name+=","
    return name


def store_seed_to_folder(seed, folder, who_called_me):
    filepath = folder+"/seeds.txt"
    with open(filepath, "a") as f:
        f.write(who_called_me + " ran with seed " + str(seed))
