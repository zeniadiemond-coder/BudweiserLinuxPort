#pragma once

#include <functional>

namespace openv6 {

void parallel_for(int count, const std::function<void(int, int)>& body);

}
