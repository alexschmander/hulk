// The headless worker does not export B-Human's debug stopwatch stream. Keep
// module execution and debug requests intact without collecting discarded CPU
// timing samples at every representation update.
#include "Debugging/TimingManager.h"
#include "Streaming/MessageQueue.h"

struct TimingManager::Pimpl {
  MessageQueue data;
};

TimingManager::TimingManager() : prvt(new Pimpl) {}
TimingManager::~TimingManager() { delete prvt; }
void TimingManager::startTiming(const char *) {}
unsigned TimingManager::stopTiming(const char *) { return 0; }
void TimingManager::signalThreadStart() { prvt->data.clear(); }
MessageQueue &TimingManager::getData() { return prvt->data; }
