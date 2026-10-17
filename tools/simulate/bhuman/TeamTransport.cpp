// Replace the upstream UDP channel at link time. TeamMessageHandler and its
// encoding, time synchronization and send budget remain native B-Human code.
#include "CognitionInputs.h"
#include "Tools/Communication/TeamMessageChannel.h"

#include <cstring>
#include <stdexcept>

void TeamMessageChannel::startLocal(int port, unsigned localId) {
  this->port = port;
  this->localId = localId;
}
void TeamMessageChannel::start(int port) { this->port = port; }
void TeamMessageChannel::trySetWifiTarget() {}
void TeamMessageChannel::send() {
  if (out.length == 0 || out.length > sizeof(out.data))
    throw std::runtime_error("invalid outgoing native team packet");
  const auto *begin = reinterpret_cast<const unsigned char *>(out.data);
  hulk_adapter::cognition.outgoing.emplace_back(begin, begin + out.length);
}
bool TeamMessageChannel::receive() {
  auto &queue = hulk_adapter::cognition.incoming;
  if (queue.empty())
    return false;
  const auto &packet = queue.front();
  if (packet.empty() || packet.size() > sizeof(in.data))
    throw std::runtime_error("invalid incoming native team packet");
  std::memcpy(in.data, packet.data(), packet.size());
  in.length = static_cast<uint8_t>(packet.size());
  queue.pop_front();
  return true;
}
