#pragma once

#include "MotionInputs.h"
#include "Representations/Communication/GameControllerData.h"
#include "Representations/Configuration/BallSpecification.h"
#include "Representations/Infrastructure/SensorData/KeyStates.h"
#include "Representations/Modeling/BallModel.h"
#include "Representations/Modeling/GlobalOpponentsModel.h"
#include "Representations/Modeling/GlobalTeammatesModel.h"
#include "Representations/Modeling/ObstacleModel.h"
#include "Representations/Modeling/RobotPose.h"
#include "Representations/Modeling/Whistle.h"

#include <deque>
#include <vector>

namespace hulk_adapter {
struct Player {
  int number;
  bool teammate;
  bool upright;
  Pose2f pose;
};
struct CognitionInputs {
  RobotPose pose;
  BallModel ball;
  std::vector<Player> players;
  KeyStates buttons;
  Whistle whistle;
  GameControllerData gameController;
  std::deque<std::vector<unsigned char>> incoming;
  std::vector<std::vector<unsigned char>> outgoing;
};
extern CognitionInputs cognition;
} // namespace hulk_adapter

MODULE(HulkCognitionInputs, {
                                ,
                                REQUIRES(BallSpecification),
                                PROVIDES(FrameInfo),
                                PROVIDES(KeyStates),
                                PROVIDES(GameControllerData),
                                PROVIDES(Whistle),
                                PROVIDES(RobotPose),
                                PROVIDES(BallModel),
                                PROVIDES(ObstacleModel),
                                PROVIDES(GlobalOpponentsModel),
                                PROVIDES(GlobalTeammatesModel),
                            });

class HulkCognitionInputs : public HulkCognitionInputsBase {
  void update(FrameInfo &value) override { value = hulk_adapter::inputs.frame; }
  void update(KeyStates &value) override {
    value = hulk_adapter::cognition.buttons;
  }
  void update(GameControllerData &value) override {
    value = hulk_adapter::cognition.gameController;
  }
  void update(Whistle &value) override {
    value = hulk_adapter::cognition.whistle;
  }
  void update(RobotPose &value) override {
    value = hulk_adapter::cognition.pose;
  }
  void update(BallModel &value) override {
    value = hulk_adapter::cognition.ball;
    value.estimate.radius = theBallSpecification.radius;
  }
  void update(ObstacleModel &value) override;
  void update(GlobalOpponentsModel &value) override;
  void update(GlobalTeammatesModel &value) override;
};
