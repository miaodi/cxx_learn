#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

static_assert(std::numeric_limits<float>::is_iec559);
static_assert(std::numeric_limits<double>::is_iec559);

struct Snapshot {
  std::string_view label;
  std::uint64_t steps;
};

void print_error_row(const Snapshot snapshot, float accumulated_float,
                     double accumulated_double, float dt_float) {
  const long double exact = static_cast<long double>(snapshot.steps) / 60.0L;
  const float from_step_count = static_cast<float>(snapshot.steps) * dt_float;

  const long double float_error = static_cast<long double>(accumulated_float) - exact;
  const long double step_count_error = static_cast<long double>(from_step_count) - exact;
  const long double double_error = static_cast<long double>(accumulated_double) - exact;

  std::cout << std::left << std::setw(10) << snapshot.label << std::right
            << std::setw(10) << snapshot.steps << std::setw(16)
            << std::setprecision(std::numeric_limits<float>::max_digits10)
            << accumulated_float << std::setw(15)
            << std::scientific << std::setprecision(3) << float_error
            << std::defaultfloat << std::setw(16)
            << std::setprecision(std::numeric_limits<float>::max_digits10)
            << from_step_count << std::setw(15) << std::scientific
            << std::setprecision(3) << step_count_error << std::defaultfloat
            << std::setw(20)
            << std::setprecision(std::numeric_limits<double>::max_digits10)
            << accumulated_double
            << std::setw(15) << std::scientific << std::setprecision(3)
            << double_error << std::defaultfloat << '\n';
}

int main() {
  constexpr std::array snapshots{
      Snapshot{"1 sec", 60},        Snapshot{"1 min", 60 * 60},
      Snapshot{"10 min", 10 * 60 * 60}, Snapshot{"1 hour", 60 * 60 * 60},
      Snapshot{"6 hours", 6 * 60 * 60 * 60},
      Snapshot{"24 hours", 24 * 60 * 60 * 60},
  };

  const float dt_float = 1.0f / 60.0f;
  const double dt_double = 1.0 / 60.0;

  std::cout << "Fixed time step: 1/60 second\n";
  std::cout << "float dt  = "
            << std::setprecision(std::numeric_limits<float>::max_digits10)
            << dt_float << '\n';
  std::cout << "double dt = "
            << std::setprecision(std::numeric_limits<double>::max_digits10)
            << dt_double << "\n\n";

  std::cout << "Repeated time += dt accumulates rounding error.\n";
  std::cout << "Computing time from an integer step count avoids the long chain of rounded additions.\n\n";

  std::cout << std::left << std::setw(10) << "duration" << std::right
            << std::setw(10) << "steps" << std::setw(16) << "float += dt"
            << std::setw(15) << "float err" << std::setw(16) << "steps*dt"
            << std::setw(15) << "steps err" << std::setw(20) << "double += dt"
            << std::setw(15) << "double err" << '\n';

  float accumulated_float = 0.0f;
  double accumulated_double = 0.0;
  std::uint64_t previous_steps = 0;

  for (const Snapshot snapshot : snapshots) {
    for (std::uint64_t step = previous_steps; step < snapshot.steps; ++step) {
      accumulated_float += dt_float;
      accumulated_double += dt_double;
    }

    print_error_row(snapshot, accumulated_float, accumulated_double, dt_float);
    previous_steps = snapshot.steps;
  }
}
