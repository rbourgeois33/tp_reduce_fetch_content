#pragma once

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

void baseline_reduce(rmm::device_uvector<int>& buffer,
                     rmm::device_scalar<int>& total);

void base(rmm::device_uvector<int>& buffer,
          rmm::device_scalar<int>& total);

void less_warp_divergence(rmm::device_uvector<int>& buffer,
          rmm::device_scalar<int>& total);

void no_bank_conflict(rmm::device_uvector<int>& buffer,
          rmm::device_scalar<int>& total);

void more_work_per_thread(rmm::device_uvector<int>& buffer,
          rmm::device_scalar<int>& total);

void unroll_last_warp(rmm::device_uvector<int>& buffer,
          rmm::device_scalar<int>& total);

void unroll_everything(rmm::device_uvector<int>& buffer,
          rmm::device_scalar<int>& total);

void cascading(rmm::device_uvector<int>& buffer,
          rmm::device_scalar<int>& total);

void better_warp_reduce(rmm::device_uvector<int>& buffer,
          rmm::device_scalar<int>& total);