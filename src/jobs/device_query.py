from subprocess import check_output, CalledProcessError
from time import sleep
import platform

system = platform.system()


def is_type(value, type_class):
    try:
        type_class(value)
        return True
    except ValueError:
        return False


def get_indentation(string: str) -> int:
    indent = 0
    for char in string:
        if char != " ":
            break
        indent += 1
    return indent


def format_string_to_key_value(string) -> tuple:
    raw_data = string.strip().split(":")
    title = raw_data[0].strip()
    data = raw_data[1:]
    if len(raw_data[1:]) > 1:
        data = ":".join(raw_data[1:]).strip()
    elif not data:
        return title, {}
    else:
        data = data[0].strip()
        if is_type(data, int):
            data = int(data)
        elif is_type(data, float):
            data = float(data)
        # else: data is string
    return title, data


def get_output(id):
    command = None
    if system == 'Linux':
        command = "nvidia-smi"
    if system == 'Windows':
        command = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi"
    if command:
        args = [command, "-q"]
        if not id is None:
            id_str = "--id={}".format(id)
            args += ([id_str])
        try:
            return check_output(args) \
                .decode("utf-8") \
                .split("\n")
        except CalledProcessError:
            return None
    return None


def get_nvidia_information(id=None) -> dict:
    nvidia_raw = get_output(id)

    if not nvidia_raw:
        return {}
    nvidia_raw = [raw for raw in nvidia_raw if (raw != '') and (not "NVSMI LOG" in raw)]

    # Finding the initial metadata:
    output = {}
    for i in range(3):
        key, value = format_string_to_key_value(nvidia_raw[i])
        output[key] = value

    # Indent manager manages where in the output dict we are. It keeps
    # keys of the current indentation level to manage where to place data.
    indent_manager = []

    for line in nvidia_raw[3:]:
        # Finds level of indentation:
        indent = int(get_indentation(line) / 4)

        key, value = format_string_to_key_value(line)

        if len(indent_manager) == 0:
            # Special case, only occurs on each GPU connected to the system.
            indent_manager = [line]
            output[line] = {}
            continue
        elif len(indent_manager) - 1 < indent:
            # Indentation increase, append key
            indent_manager += [key]
        elif len(indent_manager) - 1 > indent:
            # Indentation decrease, overwrite key and use the this key in the future:
            indent_manager[indent] = key
        else:
            # Current indentation, overwrite key in manager
            indent_manager[indent] = key

        # Find the correct placement of the current key:
        add_to = output
        for dictionary in indent_manager[:indent]:
            add_to = add_to[dictionary]

        # Special case, current element has both child data and a value:
        # Create a dict with the value placed within: key = {"value": value, "child-key": child-value}
        if isinstance(add_to, str) or isinstance(add_to, float) or isinstance(add_to, int):
            temp = add_to
            add_to = output
            for dictionary in indent_manager[:indent - 1]:
                add_to = add_to[dictionary]
            add_to[indent_manager[indent - 1]] = {"value": temp}
            add_to = output
            for dictionary in indent_manager[:indent]:
                add_to = add_to[dictionary]

        # Placement found, add data:
        add_to[key] = value

    # Turning GPUs into a list of GPUS instead of keys:
    output['GPUs'] = []
    delete_keys = []
    for key, value in output.items():
        if "GPU" in key and isinstance(value, dict):
            value["id"] = key
            output["GPUs"] += [value]
            delete_keys += [key]
            if "Processes" in value and value["Processes"] != "None":
                value['old_processes'] = value['Processes']
                value['Processes'] = []
                for process in value["old_processes"].values():
                    process['Process ID'] = process['value']
                    del process['value']
                    value['Processes'] += [process]
                del value['old_processes']
            else:
                value['Processes'] = []
    # Delete original dicts do avoid duplicates (CLEAN UP)
    for key in delete_keys:
        del output[key]

    return output


def get_least_used_gpu(preferred):
    gpu_status = []
    for id in range(0, 1000):
        info = get_nvidia_information(id)
        if not info: break
        gpu_status += [(id, info)]
    id = -1
    best = {}
    procs = 0
    try:
        for gpu_id, status in gpu_status:
            if not best:
                best, id = status, gpu_id
            else:
                procs = max(procs, len(status['GPUs'][0]['Processes']), len(best['GPUs'][0]['Processes']))
                if len(status['GPUs'][0]['Processes']) < len(best['GPUs'][0]['Processes']):
                    best, id = status, gpu_id
                elif len(status['GPUs'][0]['Processes']) == len(best['GPUs'][0]['Processes']):
                    if status['GPUs'][0]['Utilization']['Memory'] < best['GPUs'][0]['Utilization']['Memory']:
                        best, id = status, gpu_id
    except Exception:
        for gpu_id, status in gpu_status:
            if not best:
                best, id = status, gpu_id
            else:
                procs = max(procs, len(status['GPUs'][0]['Processes']), len(best['GPUs'][0]['Processes']))
                if len(status['GPUs'][1]['Processes']) < len(best['GPUs'][1]['Processes']):
                    best, id = status, gpu_id
                elif len(status['GPUs'][1]['Processes']) == len(best['GPUs'][1]['Processes']):
                    if status['GPUs'][1]['Utilization']['Memory'] < best['GPUs'][1]['Utilization']['Memory']:
                        best, id = status, gpu_id

    return id if procs > 0 else preferred


if __name__ == '__main__':
    print(f"Best gpu to use: /GPU:{get_least_used_gpu()}")
