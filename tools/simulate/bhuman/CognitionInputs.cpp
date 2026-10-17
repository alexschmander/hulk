#include "CognitionInputs.h"

namespace hulk_adapter {
CognitionInputs cognition;
}

MAKE_MODULE(HulkCognitionInputs);

void HulkCognitionInputs::update(ObstacleModel &value) {
  value.obstacles.clear();
  for (const auto &player : hulk_adapter::cognition.players) {
    const auto center =
        hulk_adapter::cognition.pose.inverse() * player.pose.translation;
    const auto type =
        player.teammate
            ? (player.upright ? Obstacle::teammate : Obstacle::fallenTeammate)
            : (player.upright ? Obstacle::opponent : Obstacle::fallenOpponent);
    value.obstacles.emplace_back(Matrix2f::Identity(), center, 200.f,
                                 hulk_adapter::inputs.frame.time, type);
    value.obstacles.back().setLeftRight(200.f);
  }
}

void HulkCognitionInputs::update(GlobalOpponentsModel &value) {
  value.opponents.clear();
  value.numOfPenalizedOpponents = 0;
  value.numOfUnknownOpponents = 0;
  for (const auto &player : hulk_adapter::cognition.players) {
    if (player.teammate)
      continue;
    GlobalOpponentsModel::OpponentEstimate opponent;
    opponent.position = player.pose.translation;
    opponent.unidentified = false;
    value.opponents.push_back(opponent);
  }
}

void HulkCognitionInputs::update(GlobalTeammatesModel &value) {
  value.teammates.clear();
  for (const auto &player : hulk_adapter::cognition.players) {
    if (!player.teammate)
      continue;
    GlobalTeammatesModel::TeammateEstimate teammate;
    teammate.playerNumber = player.number;
    teammate.pose = player.pose;
    teammate.speed = 0.f;
    value.teammates.push_back(teammate);
  }
  std::sort(value.teammates.begin(), value.teammates.end(),
            [](const auto &a, const auto &b) {
              return a.playerNumber < b.playerNumber;
            });
}
