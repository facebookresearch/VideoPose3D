% Copyright (c) 2018-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
%

function [] = ConvertDataset()

    N_JOINTS = 15; % Set to 20 if you want to export a 20-joint skeleton

    function [pose_out] = ExtractPose15(pose, dimensions)
        % We use the same 15-joint skeleton as in the evaluation
        % script "@body_pose/error.m". Proximal and Distal joints
        % are averaged.
        pose_out = NaN(15, dimensions);
        pose_out(1, :) = pose.torsoDistal; % Pelvis (root)
        pose_out(2, :) = (pose.torsoProximal + pose.headProximal) / 2; % Thorax
        pose_out(3, :) = pose.upperLArmProximal; % Left shoulder
        pose_out(4, :) = (pose.upperLArmDistal + pose.lowerLArmProximal) / 2; % Left elbow
        pose_out(5, :) = pose.lowerLArmDistal; % Left wrist
        pose_out(6, :) = pose.upperRArmProximal; % Right shoulder 
        pose_out(7, :) = (pose.upperRArmDistal + pose.lowerRArmProximal) / 2; % Right elbow
        pose_out(8, :) = pose.lowerRArmDistal; % Right wrist
        pose_out(9, :) = pose.upperLLegProximal; % Left hip
        pose_out(10, :) = (pose.upperLLegDistal + pose.lowerLLegProximal) / 2; % Left knee
        pose_out(11, :) = pose.lowerLLegDistal; % Left ankle 
        pose_out(12, :) = pose.upperRLegProximal; % Right hip
        pose_out(13, :) = (pose.upperRLegDistal + pose.lowerRLegProximal) / 2; % Right knee
        pose_out(14, :) = pose.lowerRLegDistal; % Right ankle 
        pose_out(15, :) = pose.headDistal; % Head
    end

    function [pose_out] = ExtractPose20(pose, dimensions)
        pose_out = NaN(20, dimensions);
        pose_out(1, :) = pose.torsoDistal; % Pelvis (root)
        pose_out(2, :) = pose.torsoProximal;
        pose_out(3, :) = pose.headProximal;
        pose_out(4, :) = pose.upperLArmProximal; % Left shoulder
        pose_out(5, :) = pose.upperLArmDistal;
        pose_out(6, :) = pose.lowerLArmProximal;
        pose_out(7, :) = pose.lowerLArmDistal; % Left wrist
        pose_out(8, :) = pose.upperRArmProximal; % Right shoulder 
        pose_out(9, :) = pose.upperRArmDistal;
        pose_out(10, :) = pose.lowerRArmProximal;
        pose_out(11, :) = pose.lowerRArmDistal; % Right wrist
        pose_out(12, :) = pose.upperLLegProximal; % Left hip
        pose_out(13, :) = pose.upperLLegDistal;
        pose_out(14, :) = pose.lowerLLegProximal;
        pose_out(15, :) = pose.lowerLLegDistal; % Left ankle 
        pose_out(16, :) = pose.upperRLegProximal; % Right hip
        pose_out(17, :) = pose.upperRLegDistal;
        pose_out(18, :) = pose.lowerRLegProximal;
        pose_out(19, :) = pose.lowerRLegDistal; % Right ankle 
        pose_out(20, :) = pose.headDistal; % Head
    end

    addpath('./TOOLBOX_calib/');
    addpath('./TOOLBOX_common/');
    addpath('./TOOLBOX_dxAvi/');
    addpath('./TOOLBOX_readc3d/'); 

    % Create the output directory for the converted dataset
    OUT_DIR = ['./converted_', int2str(N_JOINTS), 'j'];
    warning('off', 'MATLAB:MKDIR:DirectoryExists');
    mkdir(OUT_DIR);

    % We use the validation set as the test set
    for SPLIT = {'Train', 'Validate'}
        mkdir([OUT_DIR, '/', SPLIT{1}]);
        CurrentDataset = he_dataset('HumanEvaI', SPLIT{1});

        for SEQ = 1:length(CurrentDataset)

            Subject = char(get(CurrentDataset(SEQ), 'SubjectName'));
            Action = char(get(CurrentDataset(SEQ), 'ActionType'));
            Trial = char(get(CurrentDataset(SEQ), 'Trial'));
            DatasetBasePath = char(get(CurrentDataset(SEQ), 'DatasetBasePath'));
            if Trial ~= '1'
                % We are only interested in fully-annotated data
                continue;
            end

            if strcmp(Action, 'ThrowCatch') && strcmp(Subject, 'S3')
                % Damaged mocap stream
                continue;
            end

            fprintf('Converting...\n')
            fprintf('\tSplit: %s\n', SPLIT{1});
            fprintf('\tSubject: %s\n', Subject);
            fprintf('\tAction: %s\n', Action);    
            fprintf('\tTrial: %s\n', Trial);

            % Create subject directory if it does not exist
            mkdir([OUT_DIR, '/', SPLIT{1}, '/', Subject]);

            % Load the sequence
            [~, ~, MocapStream, MocapStream_Enabled] ...
                                    = sync_stream(CurrentDataset(SEQ));

            % Set frame range
            FrameStart = get(CurrentDataset(SEQ), 'FrameStart');
            FrameStart = [FrameStart{:}];
            FrameEnd   = get(CurrentDataset(SEQ), 'FrameEnd');    
            FrameEnd   = [FrameEnd{:}];

            fprintf('\tNum. frames: %d\n', FrameEnd - FrameStart + 1);
            poses_3d = NaN(FrameEnd - FrameStart + 1, N_JOINTS, 3);
            poses_2d = NaN(3, FrameEnd - FrameStart + 1, N_JOINTS, 2);
            corrupt = 0;
            for FRAME = FrameStart:FrameEnd

                if (MocapStream_Enabled)
                    [MocapStream, pose, ValidPose] = cur_frame(MocapStream, FRAME, 'body_pose');

                    if (ValidPose)
                        i = FRAME - FrameStart + 1;
                        
                        % Extract 3D pose
                        if N_JOINTS == 15
                            poses_3d(i, :, :) = ExtractPose15(pose, 3);
                        else
                            poses_3d(i, :, :) = ExtractPose20(pose, 3);
                        end
                        
                        % Extract ground-truth 2D pose via camera
                        % projection
                        for CAM = 1:3
                            if (CAM == 1)
                                CameraName = 'C1';
                            elseif (CAM == 2)
                                CameraName = 'C2';    
                            elseif (CAM == 3)
                                CameraName = 'C3';
                            end
                            CalibrationFilename = [DatasetBasePath, Subject, '/Calibration_Data/', CameraName,  '.cal'];
                            pose_2d = project2d(pose, CalibrationFilename);
                            if N_JOINTS == 15
                                poses_2d(CAM, i, :, :) = ExtractPose15(pose_2d, 2);
                            else
                                poses_2d(CAM, i, :, :) = ExtractPose20(pose_2d, 2);
                            end
                        end
                        
                    else
                        corrupt = corrupt + 1;
                    end
                end
            end
            fprintf('\n%d out of %d frames are damaged\n', corrupt, FrameEnd - FrameStart + 1);
            FileName = [OUT_DIR, '/', SPLIT{1}, '/', Subject, '/', Action, '_', Trial, '.mat'];
            save(FileName, 'poses_3d', 'poses_2d');
            fprintf('... saved to %s\n\n', FileName);
        end
    end
end
