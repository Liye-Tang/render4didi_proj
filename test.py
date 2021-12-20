# import pyrosbag
#
# bag_file = 'data.bag'
# bag = pyrosbag.Bag(bag_file)
# bag_data = bag.read_messages()
# print(bag_data)

import struct
import os
from protobuf_to_dict import protobuf_to_dict
# import idc_planner_info_pb2 as pb
import idc_info_pb2 as pb2


# search_service.ParseFromString(b)
# # print(search_service.type)
# d = protobuf_to_dict(search_service)
# print(d, type(d))

if __name__ == '__main__':
    path = '/home/tly/Desktop/plot_for_replay/test/test_20211217_2pm/exp_2021_12_17_15_12_16/try_2021_12_17_15_17_15'  # 文件夹目录
    filepaths = os.listdir(path)  # 得到文件夹下的所有文件名称
    # print(filepaths)
    idc_planner_info_list = []
    for filepath in filepaths:
        binfile = open(path+'/'+filepath, 'rb')  # 打开二进制文件
        size = os.path.getsize(path+'/'+filepath)  # 获得文件大小
        data = binfile.read(size)  # 每次输出一个字节
        binfile.close()
        IdcPlannerInfo = pb2.IdcPlannerInfo()
        IdcPlannerInfo.ParseFromString(data)
        d = protobuf_to_dict(IdcPlannerInfo)
        idc_planner_info_list.append(d)
        # print(d.keys())
        print(d['traj_pose'][-1])

    # print(IdcPlannerInfo.decison)


    # print(search_service.type)
    # d = protobuf_to_dict(search_service)
    # print(d, type(d))
