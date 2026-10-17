#pragma once

#include "Framework/Module.h"
#include "Representations/Infrastructure/FrameInfo.h"
#include "Representations/Infrastructure/RobotHealth.h"
#include "Representations/Infrastructure/SensorData/InertialSensorData.h"
#include "Representations/Infrastructure/SensorData/JointSensorData.h"
#include "Representations/MotionControl/HeadMotionRequest.h"
#include "Representations/MotionControl/MotionRequest.h"
#include "Representations/Sensing/GyroOffset.h"

#include <array>

namespace hulk_adapter {
constexpr std::array<Joints::Joint, 22> serialJoints = {
    Joints::headYaw,        Joints::headPitch,     Joints::lShoulderPitch,
    Joints::lShoulderRoll,  Joints::lElbowPitch,   Joints::lElbowYaw,
    Joints::rShoulderPitch, Joints::rShoulderRoll, Joints::rElbowPitch,
    Joints::rElbowYaw,      Joints::lHipPitch,     Joints::lHipRoll,
    Joints::lHipYaw,        Joints::lKneePitch,    Joints::lAnklePitch,
    Joints::lAnkleRoll,     Joints::rHipPitch,     Joints::rHipRoll,
    Joints::rHipYaw,        Joints::rKneePitch,    Joints::rAnklePitch,
    Joints::rAnkleRoll};

struct Inputs {
  FrameInfo frame;
  JointSensorData joints;
  InertialSensorData inertial;
  MotionRequest motion;
};
// The worker owns exactly one robot, and executes its graph on the IPC thread.
extern Inputs inputs;
} // namespace hulk_adapter

MODULE(HulkMotionInputs, {
                             ,
                             PROVIDES(FrameInfo),
                             PROVIDES(JointSensorData),
                             PROVIDES(InertialSensorData),
                             PROVIDES(MotionRequest),
                             PROVIDES(HeadMotionRequest),
                             PROVIDES(RobotHealth),
                             PROVIDES(GyroOffset),
                         });

class HulkMotionInputs : public HulkMotionInputsBase {
  void update(FrameInfo &value) override { value = hulk_adapter::inputs.frame; }
  void update(JointSensorData &value) override {
    value = hulk_adapter::inputs.joints;
  }
  void update(InertialSensorData &value) override {
    value = hulk_adapter::inputs.inertial;
  }
  void update(MotionRequest &value) override {
    value = hulk_adapter::inputs.motion;
  }
  void update(HeadMotionRequest &value) override {
    value = HeadMotionRequest();
  }
  void update(RobotHealth &value) override {
    value.olderPerceptionTimestamp = hulk_adapter::inputs.frame.time;
  }
  void update(GyroOffset &value) override {
    value.offsetCheckFinished = true;
    value.isIMUBad = false;
  }
};
