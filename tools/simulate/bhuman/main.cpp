#include "CognitionInputs.h"
#include "Framework/ModuleGraphCreator.h"
#include "Framework/ModuleGraphRunner.h"
#include "Framework/ThreadFrame.h"
#include "Platform/File.h"
#include "Platform/Time.h"
#include "Representations/Communication/BHumanMessageOutputGenerator.h"
#include "Representations/Communication/TeamData.h"
#include "Representations/Configuration/JointCalibration.h"
#include "Representations/Configuration/MotorParameters.h"
#include "Representations/Infrastructure/GameState.h"
#include "Representations/Infrastructure/JointRequest.h"
#include "Representations/Infrastructure/LEDRequest.h"
#include "Representations/MotionControl/MotionInfo.h"
#include "Streaming/FunctionList.h"
#include "Streaming/InStreams.h"
#include "Streaming/OutStreams.h"
#include "protocol_generated.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace wire = hulk::bhuman;
constexpr uint32_t maxMessage = 1024 * 1024;

STREAMABLE(ActuatorSettings, {
                                 ,
                                 (Angle)startHeadPitchTorque,
                                 (float)headPitchTorqueDiff,
                             });

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void receiveAll(int fd, void *buffer, size_t length) {
  auto *bytes = static_cast<unsigned char *>(buffer);
  while (length) {
    const ssize_t count = read(fd, bytes, length);
    require(count > 0, "worker socket closed or read failed");
    bytes += count;
    length -= static_cast<size_t>(count);
  }
}

void sendAll(int fd, const flatbuffers::FlatBufferBuilder &builder) {
  const auto *bytes = builder.GetBufferPointer();
  size_t remaining = builder.GetSize();
  while (remaining) {
    const ssize_t count = send(fd, bytes, remaining, MSG_NOSIGNAL);
    require(count > 0, "worker socket write failed");
    bytes += count;
    remaining -= static_cast<size_t>(count);
  }
}

class ModuleRuntime : public ThreadFrame {
  std::unique_ptr<ModuleGraphRunner> graph;

public:
  explicit ModuleRuntime(const Settings &settings)
      : ThreadFrame(settings, "BHuman", nullptr, nullptr) {
    setGlobals();
  }
  ~ModuleRuntime() override {
    setGlobals();
    graph.reset();
  }
  void configure(ModuleGraphCreator &creator, size_t index, size_t count) {
    setGlobals();
    OutBinaryMemory execution;
    execution << creator.getExecutionValues(index) << 1u;
    graph = std::make_unique<ModuleGraphRunner>(count);
    InBinaryMemory input(execution.data());
    graph->update(input);
  }
  void execute() {
    setGlobals();
    graph->execute();
    debugSender->clear();
  }
  void transfer(ModuleRuntime &receiver, size_t destination, size_t source) {
    setGlobals();
    OutBinaryMemory packet;
    graph->writePacket(packet, destination);
    receiver.setGlobals();
    InBinaryMemory input(packet.data());
    receiver.graph->readPacket(input, source);
  }
  template <typename T> const T &get(const char *name) {
    setGlobals();
    return static_cast<const T &>(Blackboard::getInstance()[name]);
  }

private:
  int getPriority() const override { return 0; }
  void init() override {}
  bool main() override { return true; }
  void terminate() override {}
};

class RobotRuntime {
  ModuleRuntime motion;
  std::unique_ptr<ModuleRuntime> cognition;
  ActuatorSettings actuators;
  uint64_t cognitionTime = 0;

public:
  RobotRuntime(const Settings &settings, bool soccer, bool active)
      : motion(settings) {
    InMapFile actuatorConfig("boosterProvider.cfg",
                             ~bit(InMap::unusedAttribute));
    require(actuatorConfig.exists(), "missing native actuator configuration");
    actuatorConfig >> actuators;
    if (soccer) {
      File strategy("Behavior/Strategies/s5v5.cfg", "r");
      require(strategy.exists(),
              "missing B-Human behavior configuration; rerun build.py");
    }
    Configuration original;
    InMapFile input(std::string(File::getBHDir()) +
                    "/Config/Scenarios/Default/K1/threads.cfg");
    require(input.exists(), "missing upstream K1 threads.cfg");
    input >> original;
    Configuration config;
    config.defaultRepresentations = {
        "CameraCalibration", "CameraCalibrationStatus",
        "CameraInfo",        "CameraMatrix",
        "RobotPercept",      "EmergencyButtonState",
        "DampingState",      "JoystickState",
        "SystemSensorData"};
    for (const char *name : {"Motion", "Cognition"}) {
      if (!soccer && std::string(name) == "Cognition")
        continue;
      const auto thread =
          std::find_if(original().begin(), original().end(),
                       [name](const auto &t) { return t.name == name; });
      require(thread != original().end(), "missing upstream module graph");
      config().push_back(*thread);
    }
    auto &motionProviders = config()[0].representationProviders;
    std::erase_if(motionProviders, [](const auto &p) {
      return p.provider == "BoosterProvider" ||
             p.representation == "InternalState" ||
             p.representation == "GyroOffset";
    });
    for (const char *name :
         {"FrameInfo", "JointSensorData", "InertialSensorData", "GyroOffset"})
      motionProviders.emplace_back(name, "HulkMotionInputs");
    if (!soccer) {
      config.defaultRepresentations.insert(
          config.defaultRepresentations.end(),
          {"ArmMotionRequest", "CalibrationRequest", "FieldInterceptBall",
           "GameState", "KeyStates", "RobotPose"});
      for (const char *name :
           {"MotionRequest", "HeadMotionRequest", "RobotHealth"})
        motionProviders.emplace_back(name, "HulkMotionInputs");
    } else {
      auto &providers = config()[1].representationProviders;
      const std::vector<std::string> replaced = {"ColorFrameInfoProvider",
                                                 "MotionProvider",
                                                 "SelfLocator",
                                                 "AlternativeRobotPoseProvider",
                                                 "BallStateEstimator",
                                                 "BallPerceptFilter",
                                                 "PerceptRegistrationProvider",
                                                 "ObstacleModelProvider",
                                                 "GlobalOpponentsTracker",
                                                 "GlobalTeammatesTracker",
                                                 "GameControllerDataProvider",
                                                 "IMUCalibrationProvider",
                                                 "StaticInitialPoseProvider",
                                                 "SideInformationProvider"};
      std::erase_if(providers, [&replaced](const auto &p) {
        return std::find(replaced.begin(), replaced.end(), p.provider) !=
               replaced.end();
      });
      for (const char *name :
           {"FrameInfo", "KeyStates", "GameControllerData", "Whistle",
            "RobotPose", "BallModel", "ObstacleModel", "GlobalOpponentsModel",
            "GlobalTeammatesModel"})
        providers.emplace_back(name, "HulkCognitionInputs");
      config.defaultRepresentations.insert(config.defaultRepresentations.end(),
                                           {"AlternativeRobotPoseHypothesis",
                                            "StaticInitialPose",
                                            "SideInformation"});
    }
    ModuleGraphCreator creator(config);
    OutBinaryMemory serialized;
    serialized << config;
    InBinaryMemory configInput(serialized.data());
    require(creator.update(configInput), "invalid headless robot module graph");
    motion.configure(creator, 0, config().size());
    if (soccer) {
      cognition = std::make_unique<ModuleRuntime>(settings);
      cognition->configure(creator, 1, 2);
      if (active)
        Global::getDebugRequestTable().addRequest(
            DebugRequest("module:GameStateProvider:active"));
    }
  }
  std::vector<wire::JointCommand> step(const wire::Step &step) {
    motion.setGlobals();
    require(step.joints()->size() == 22, "expected 22 serial joint samples");
    require(step.angle() && step.gyro() && step.acceleration() &&
                step.velocity(),
            "missing sensor vectors");
    auto &inputs = hulk_adapter::inputs;
    inputs.frame.time =
        100000 + static_cast<unsigned>(step.time_ns() / 1000000);
    Time::addSimulatedTime(
        static_cast<int>(inputs.frame.time - Time::getCurrentSystemTime()));
    inputs.joints.timestamp = inputs.frame.time;
    for (size_t i = 0; i < hulk_adapter::serialJoints.size(); ++i) {
      const auto joint = hulk_adapter::serialJoints[i];
      const auto &sample =
          *step.joints()->Get(static_cast<flatbuffers::uoffset_t>(i));
      require(std::isfinite(sample.position()) &&
                  std::isfinite(sample.velocity()) &&
                  std::isfinite(sample.torque()),
              "non-finite joint sensor");
      inputs.joints.angles[joint] = sample.position();
      inputs.joints.velocity[joint] = sample.velocity();
      inputs.joints.torque[joint] = sample.torque();
    }
    const auto vector = [](const wire::Vector3 *v) {
      require(std::isfinite(v->x()) && std::isfinite(v->y()) &&
                  std::isfinite(v->z()),
              "non-finite vector");
      return Vector3f(v->x(), v->y(), v->z());
    };
    inputs.inertial.angle = vector(step.angle()).cast<Angle>();
    inputs.inertial.gyro = vector(step.gyro()).cast<Angle>();
    inputs.inertial.acc = vector(step.acceleration());
    inputs.motion = MotionRequest();
    switch (step.motion()) {
    case wire::Motion::Stand:
      inputs.motion.motion = MotionRequest::stand;
      break;
    case wire::Motion::Walk:
      inputs.motion.motion = MotionRequest::walkAtAbsoluteSpeed;
      break;
    case wire::Motion::Prepare:
      inputs.motion.motion = MotionRequest::prepare;
      break;
    case wire::Motion::Damping:
      inputs.motion.motion = MotionRequest::playDead;
      break;
    default:
      throw std::runtime_error("unsupported motion request");
    }
    const Vector3f velocity = vector(step.velocity());
    inputs.motion.walkSpeed =
        Pose2f(velocity.z(), velocity.x() * 1000.f, velocity.y() * 1000.f);
    if (cognition)
      readCognition(step);
    motion.execute();
    if (cognition && step.time_ns() % 20000000 == 0) {
      motion.transfer(*cognition, 1, 0);
      cognition->execute();
      const auto &generator = cognition->get<BHumanMessageOutputGenerator>(
          "BHumanMessageOutputGenerator");
      if (generator.send && generator.sendThisFrame &&
          generator.sendThisFrame())
        generator.send();
      cognition->transfer(motion, 0, 1);
      cognitionTime = step.time_ns();
    }
    const auto &request = motion.get<JointRequest>("JointRequest");
    const auto &motors = motion.get<MotorParameters>("MotorParameters");
    const auto &calibration = motion.get<JointCalibration>("JointCalibration");
    std::vector<wire::JointCommand> commands;
    for (const auto joint : hulk_adapter::serialJoints) {
      const auto &gains =
          motors.controlParameters[request.jointStiffnessMapping][joint];
      const int percentage = request.stiffnessData.stiffnesses[joint];
      require(percentage >= 0 && percentage <= 100,
              "unresolved joint stiffness");
      const float stiffness = percentage / 100.f;
      const bool off =
          request.angles[joint] == JointAngles::off || percentage == 0;
      require(request.angles[joint] != JointAngles::ignore,
              "unresolved joint ignore request");
      const float position =
          off ? static_cast<float>(inputs.joints.angles[joint])
              : static_cast<float>(request.angles[joint] +
                                   calibration.offsets[joint]);
      const bool torqueControl = request.isTorqueControl &&
                                 joint != Joints::headYaw &&
                                 joint != Joints::headPitch;
      float torque = 0.f;
      if (!off && torqueControl)
        torque = std::clamp((position - inputs.joints.angles[joint]) * gains.p *
                                stiffness,
                            -gains.maxTorque, gains.maxTorque);
      else if (!off && joint == Joints::headPitch)
        torque = std::clamp(
            std::max(0.f, position - static_cast<float>(
                                         actuators.startHeadPitchTorque)) *
                actuators.headPitchTorqueDiff * gains.p * stiffness,
            -gains.maxTorque * stiffness, gains.maxTorque * stiffness);
      commands.emplace_back(
          torqueControl ? static_cast<float>(inputs.joints.angles[joint])
                        : position,
          0.f, torque, off || torqueControl ? 0.f : stiffness * gains.p,
          off ? motors.dDamping[joint] : stiffness * gains.d, gains.maxTorque);
    }
    return commands;
  }
  flatbuffers::Offset<wire::Commands>
  reply(flatbuffers::FlatBufferBuilder &builder,
        const std::vector<wire::JointCommand> &joints) {
    const auto phase = TypeRegistry::getEnumName(
        motion.get<MotionInfo>("MotionInfo").executedPhase);
    const auto requested = TypeRegistry::getEnumName(
        motion.get<MotionRequest>("MotionRequest").motion);
    const auto &state = cognition ? cognition->get<GameState>("GameState")
                                  : motion.get<GameState>("GameState");
    const auto game = TypeRegistry::getEnumName(state.state);
    const auto player = TypeRegistry::getEnumName(state.playerState);
    LEDRequest leds;
    size_t teammates = 0;
    if (cognition) {
      leds = cognition->get<LEDRequest>("LEDRequest");
      teammates = cognition->get<TeamData>("TeamData").teammates.size();
    }
    const wire::Leds colors(
        static_cast<wire::LedState>(leds.ledStates[LEDRequest::red]),
        static_cast<wire::LedState>(leds.ledStates[LEDRequest::green]),
        static_cast<wire::LedState>(leds.ledStates[LEDRequest::blue]));
    std::vector<flatbuffers::Offset<wire::Packet>> packets;
    for (const auto &packet : hulk_adapter::cognition.outgoing)
      packets.push_back(
          wire::CreatePacket(builder, builder.CreateVector(packet)));
    hulk_adapter::cognition.outgoing.clear();
    return wire::CreateCommands(
        builder, builder.CreateVectorOfStructs(joints),
        builder.CreateString(phase), builder.CreateString(requested),
        builder.CreateString(game), builder.CreateString(player), &colors,
        builder.CreateVector(packets), static_cast<uint8_t>(teammates),
        cognitionTime);
  }

private:
  void readCognition(const wire::Step &step) {
    auto &data = hulk_adapter::cognition;
    const unsigned time = hulk_adapter::inputs.frame.time;
    const auto pose = [](const wire::Pose2 *p) {
      require(p && std::isfinite(p->x()) && std::isfinite(p->y()) &&
                  std::isfinite(p->yaw()),
              "invalid world pose");
      return Pose2f(p->yaw(), p->x() * 1000.f, p->y() * 1000.f);
    };
    require(step.world(), "cognition requires world ground truth");
    data.pose = pose(step.world()->pose());
    data.pose.quality = RobotPose::superb;
    if (const auto *ball = step.world()->ball()) {
      require(ball->position() && ball->velocity(),
              "invalid ball ground truth");
      const auto *p = ball->position();
      const auto *v = ball->velocity();
      require(std::isfinite(p->x()) && std::isfinite(p->y()) &&
                  std::isfinite(v->x()) && std::isfinite(v->y()),
              "invalid ball vector");
      data.ball.estimate.position =
          data.pose.inverse() * Vector2f(p->x() * 1000.f, p->y() * 1000.f);
      data.ball.estimate.velocity = Vector2f(v->x() * 1000.f, v->y() * 1000.f)
                                        .rotated(-data.pose.rotation);
      data.ball.lastPerception = data.ball.estimate.position;
      data.ball.timeWhenLastSeen = data.ball.timeWhenDisappeared = time;
      data.ball.seenPercentage = 100;
    } else
      data.ball.seenPercentage = 0;
    data.players.clear();
    if (const auto *players = step.world()->players()) {
      require(players->size() <= 9, "too many world players");
      for (const auto *p : *players) {
        require(p->number() >= 1 && p->number() <= 5,
                "invalid world player number");
        data.players.push_back(
            {p->number(), p->teammate(), p->upright(), pose(p->pose())});
      }
    }
    if (const auto *buttons = step.buttons()) {
      data.buttons.pressed[KeyStates::f1] = buttons->f1();
      data.buttons.pressed[KeyStates::stand] = buttons->stand();
      data.buttons.pressed[KeyStates::walk] = buttons->walk();
    } else
      data.buttons = KeyStates();
    if (const auto *whistle = step.whistle()) {
      require(whistle->time_ns() <= step.time_ns() &&
                  std::isfinite(whistle->confidence()) &&
                  whistle->confidence() >= 0 && whistle->channels() > 0,
              "invalid whistle event");
      data.whistle.lastTimeWhistleDetected =
          100000 + static_cast<unsigned>(whistle->time_ns() / 1000000);
      data.whistle.confidenceOfLastWhistleDetection = whistle->confidence();
      data.whistle.channelsUsedForWhistleDetection = whistle->channels();
    }
    // The simulated microphone is always listening, without running an audio
    // pipeline.
    data.whistle.channelsUsedForWhistleDetection = 2;
    if (const auto *packet = step.game_controller()) {
      static_assert(sizeof(RoboCup::RoboCupGameControlData) == 158);
      require(packet->bytes()->size() == 158 &&
                  std::memcmp(packet->bytes()->data(), "RGme", 4) == 0 &&
                  packet->bytes()->Get(4) == 20,
              "invalid GameController v20 packet");
      std::memcpy(
          static_cast<RoboCup::RoboCupGameControlData *>(&data.gameController),
          packet->bytes()->data(), 158);
      require(data.gameController.playersPerTeam >= 1 &&
                  data.gameController.playersPerTeam <= MAX_NUM_PLAYERS,
              "invalid GameController player count");
      data.gameController.timeLastPacketReceived = time;
      data.gameController.isTrueData = false;
    }
    if (const auto *packets = step.team_messages()) {
      require(data.incoming.size() + packets->size() <= 64,
              "native team packet queue overflow");
      for (const auto *packet : *packets) {
        require(packet->bytes()->size() > 0 && packet->bytes()->size() <= 128,
                "invalid native team packet");
        data.incoming.emplace_back(packet->bytes()->begin(),
                                   packet->bytes()->end());
      }
    }
  }
};

int main(int argc, char **argv) {
  if (argc != 3 || std::string(argv[1]) != "--socket") {
    std::cerr << "Usage: bhuman-motion-worker --socket PATH\n";
    return 2;
  }
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  uint64_t epoch = 0, sequence = 0;
  try {
    require(fd >= 0, "cannot create worker socket");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    require(std::strlen(argv[2]) < sizeof(address.sun_path),
            "socket path too long");
    std::strcpy(address.sun_path, argv[2]);
    require(connect(fd, reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) == 0,
            "cannot connect worker socket");
    FunctionList::execute();
    Time::initialize();
    Time::setSimulatedTime(true);
    SystemCall::mute(true);
    std::unique_ptr<RobotRuntime> runtime;
    uint64_t nextTime = 0, nextSequence = 1;
    while (true) {
      unsigned char prefix[4];
      receiveAll(fd, prefix, sizeof(prefix));
      const auto size = flatbuffers::ReadScalar<uint32_t>(prefix);
      require(size > 0 && size <= maxMessage, "invalid message length");
      std::vector<unsigned char> bytes(size + 4);
      std::memcpy(bytes.data(), prefix, 4);
      receiveAll(fd, bytes.data() + 4, size);
      flatbuffers::Verifier verifier(bytes.data(), bytes.size());
      require(wire::VerifySizePrefixedEnvelopeBuffer(verifier),
              "invalid FlatBuffers message");
      const auto &message = *wire::GetSizePrefixedEnvelope(bytes.data());
      require(message.version() == 2, "unsupported protocol version");
      sequence = message.sequence();
      flatbuffers::FlatBufferBuilder reply;
      flatbuffers::Offset<void> payload;
      wire::Payload kind;
      if (message.payload_type() == wire::Payload::Initialize) {
        require(!runtime && message.sequence() == 0 && message.epoch() != 0,
                "invalid initialization");
        epoch = message.epoch();
        const auto player = message.payload_as_Initialize()->player();
        require(player >= 1 && player <= 5, "invalid player number");
        const auto *init = message.payload_as_Initialize();
        require(init->team() > 0, "invalid team number");
        Settings settings("SimulatedK1", 0.002f, Settings::k1, init->team(),
                          Settings::white, Settings::purple, player, "Default",
                          "Default", 0);
        if (const auto *directory = init->config_directory())
          settings.searchPath.insert(settings.searchPath.begin(),
                                     directory->str() + "/");
        runtime = std::make_unique<RobotRuntime>(settings, init->cognition(),
                                                 init->active());
        std::vector<flatbuffers::Offset<flatbuffers::String>> joints;
        for (const auto joint : hulk_adapter::serialJoints)
          joints.push_back(
              reply.CreateString(TypeRegistry::getEnumName(joint)));
        payload = wire::CreateReady(reply, reply.CreateString(BHUMAN_REVISION),
                                    reply.CreateVector(joints))
                      .Union();
        kind = wire::Payload::Ready;
      } else {
        require(runtime && message.epoch() == epoch && sequence == nextSequence,
                "stale or out-of-order request");
        if (message.payload_type() == wire::Payload::Stop) {
          runtime.reset();
          payload = wire::CreateStopped(reply).Union();
          kind = wire::Payload::Stopped;
        } else {
          require(message.payload_type() == wire::Payload::Step,
                  "unexpected request");
          const auto &step = *message.payload_as_Step();
          require(step.time_ns() == nextTime,
                  "expected a consecutive 2 ms sensor frame");
          const auto commands = runtime->step(step);
          payload = runtime->reply(reply, commands).Union();
          kind = wire::Payload::Commands;
          nextTime += 2000000;
        }
        ++nextSequence;
      }
      wire::FinishSizePrefixedEnvelopeBuffer(
          reply,
          wire::CreateEnvelope(reply, 2, epoch, sequence, kind, payload));
      sendAll(fd, reply);
      if (kind == wire::Payload::Stopped)
        break;
    }
    close(fd);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B-Human worker: " << error.what() << '\n';
    if (fd >= 0) {
      flatbuffers::FlatBufferBuilder reply;
      auto payload =
          wire::CreateError(reply, reply.CreateString(error.what())).Union();
      wire::FinishSizePrefixedEnvelopeBuffer(
          reply, wire::CreateEnvelope(reply, 2, epoch, sequence,
                                      wire::Payload::Error, payload));
      try {
        sendAll(fd, reply);
      } catch (...) {
      }
      close(fd);
    }
    return 1;
  }
}

SystemCall::Mode SystemCall::getMode() { return SystemCall::simulatedRobot; }
