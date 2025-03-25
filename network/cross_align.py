import torch


def displacement(displace_scale, displacement):
    if displacement and displace_scale != [0]:
        s = displace_scale
        displacement_map_list = [[(0, 0, 0)]]

        # 生成三维位移
        for scale in s:
            displacement_map_list.append(
                [
                    (-scale, -scale, 0),
                    (scale, scale, 0),
                    (-scale, 0, 0),
                    (scale, 0, 0),
                    (0, scale, 0),
                    (0, -scale, 0),
                    (-scale, scale, 0),
                    (scale, -scale, 0),
                    (0, 0, -scale),
                    (0, 0, scale),  # 新增Z轴位移
                    (-scale, 0, -scale),
                    (scale, 0, scale),
                    (0, -scale, -scale),
                    (0, scale, scale),  # Z轴组合位移
                    (-scale, -scale, -scale),
                    (scale, scale, scale),
                    (-scale, -scale, scale),
                    (scale, scale, -scale),  # 所有轴组合
                ]
            )

        # 将所有位移列表合并到一个列表中
        for s_idx in range(1, len(s) + 1):
            for i in range(len(displacement_map_list[s_idx])):
                displacement_map_list[0].append(displacement_map_list[s_idx][i])
        displacement_map_list = displacement_map_list[0]
    else:
        displacement_map_list = [(0, 0, 0)]  # 如果没有位移，仅包含原点

    return displacement_map_list


def distance_to_similarity(distance, temperature=1.0, dim=-1, eps=1e-8):
    exp_distances = torch.exp(-distance / temperature) + eps
    similarity = exp_distances / torch.sum(exp_distances, dim, keepdim=True)
    return similarity


def compute_joint_distribution(x_out, displacement_map: (int, int, int)):
    x_out = x_out.permute(0, 1, 4, 2, 3)
    n, c, d, h, w = x_out.shape
    # print(displacement_map[0], displacement_map[1],displacement_map[2])
    after_displacement = x_out.roll(shifts=[displacement_map[0], displacement_map[1], displacement_map[2]],
                                    dims=[2, 3, 4])
    # print(after_displacement.shape)
    x_out = x_out.reshape(n, c, d * h * w)
    after_displacement = after_displacement.reshape(n, c, d * h * w).transpose(2, 1)
    p_i_j = (x_out @ after_displacement).mean(0).unsqueeze(0).unsqueeze(0)
    p_i_j += 1e-8
    p_i_j /= p_i_j.sum(dim=3, keepdim=True)  # norm
    return p_i_j.contiguous()


def compute_align_loss(cluster_s, cluster_t, displacement_map_list, align_type):
    loss = 0

    for dis_map in displacement_map_list:
        p_joint_s = compute_joint_distribution(x_out=cluster_s, displacement_map=dis_map)
        p_joint_t = compute_joint_distribution(x_out=cluster_t, displacement_map=dis_map)

        # 防止 log(0) 引发 NaN 问题
        p_joint_s = torch.clamp(p_joint_s, min=1e-10)
        p_joint_t = torch.clamp(p_joint_t, min=1e-10)

        if align_type == "js":
            m = (p_joint_s + p_joint_t) / 2
            loss += (
                    0.5
                    * torch.nn.functional.kl_div(torch.log(p_joint_t), m, reduction="none")
                    .sum(3)
                    .mean()
                    + 0.5
                    * torch.nn.functional.kl_div(torch.log(p_joint_s), m, reduction="none")
                    .sum(3)
                    .mean()
            )
        elif align_type == "kl":
            loss += (
                torch.nn.functional.kl_div(
                    torch.log(p_joint_t), p_joint_s, reduction="none"
                )
                .sum(3)
                .mean()
            )
        elif align_type == "l1":
            loss += torch.mean(torch.abs((p_joint_s - p_joint_t)))
        else:
            raise ValueError(f"Unknown align_type: {align_type}")

    return loss / len(displacement_map_list)


print(len(displacement([1,2,3],displacement=True)))