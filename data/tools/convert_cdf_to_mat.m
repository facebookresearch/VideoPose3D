% Copyright (c) 2018-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
%

% Extract "Poses_D3_Positions_S*.tgz" to the "pose" directory
% and run this script to convert all .cdf files to .mat

pose_directory = 'pose';
dirs = dir(strcat(pose_directory, '/*/MyPoseFeatures/D3_Positions/*.cdf'));

paths = {dirs.folder};
names = {dirs.name};

for i = 1:numel(names)
    data = cdfread(strcat(paths{i}, '/', names{i}));
    save(strcat(paths{i}, '/', names{i}, '.mat'), 'data');
end