#include <vector>
#include <utility>
#include <cmath>
#include <torch/extension.h>

void butterfly_factor_multiply_cuda(const at::Tensor& twiddle, const at::Tensor& input, at::Tensor& output);
void butterfly_factor_multiply_backward_cuda(const at::Tensor& grad, const at::Tensor& twiddle, const at::Tensor& input,
                                             at::Tensor& d_twiddle_expanded, at::Tensor& d_input);
void butterfly_multiply_inplace_cuda(const at::Tensor& twiddle, at::Tensor& input);
void butterfly_multiply_inplace_backward_cuda(const at::Tensor& grad, const at::Tensor& twiddle, at::Tensor& output,
                                              at::Tensor& d_twiddle, at::Tensor& d_input);
void butterfly_multiply_intermediate_cuda(const at::Tensor& twiddle, at::Tensor& input, bool increasing_stride, bool return_intermediates);
void butterfly_multiply_intermediate_backward_cuda(const at::Tensor& twiddle, const at::Tensor& output,
                                                   at::Tensor& d_twiddle, at::Tensor& d_input, bool increasing_stride);
void butterfly_multiply_untied_cuda(const at::Tensor& twiddle, at::Tensor& input, bool increasing_stride, bool return_intermediates);
void butterfly_multiply_untied_backward_cuda(const at::Tensor& twiddle, const at::Tensor& output,
                                             at::Tensor& d_twiddle, at::Tensor& d_input, bool increasing_stride);

void butterfly_conv2d_cuda(const at::Tensor& twiddle, const at::Tensor& input, at::Tensor& output,
                           const int kernel_size, const int padding, const int h_out, 
                           const int w_out, bool increasing_stride, bool return_intermediates);
void butterfly_conv2d_backward_cuda(const at::Tensor& grad, const at::Tensor& twiddle, 
                                    const at::Tensor& output, at::Tensor& d_twiddle, 
                                    at::Tensor& d_input, const int kernel_size, const int padding,
                                    const int h_out, const int w_out,
                                    bool increasing_stride);
void butterfly_multiply_untied_svd_cuda(const at::Tensor& twiddle, at::Tensor& input,
                                        bool increasing_stride, bool return_intermediates);
void butterfly_multiply_untied_svd_backward_cuda(const at::Tensor& twiddle, const at::Tensor& output,
                                                 at::Tensor& d_twiddle, at::Tensor& d_input, bool increasing_stride);
void permutation_factor_even_odd_multiply_cuda(const at::Tensor& p, const at::Tensor& input, at::Tensor& output);
void permutation_factor_even_odd_multiply_backward_cuda(const at::Tensor& grad, const at::Tensor& p, const at::Tensor& input,
                                                        at::Tensor& d_p_expanded, at::Tensor& d_input);
void permutation_factor_reverse_multiply_cuda(const at::Tensor& p, const at::Tensor& input, at::Tensor& output);
void permutation_factor_reverse_multiply_backward_cuda(const at::Tensor& grad, const at::Tensor& p, const at::Tensor& input,
                                                       at::Tensor& d_p_expanded, at::Tensor& d_input);

#define CHECK_DEVICE(x) AT_CHECK(x.type().device_type() == at::kCPU || x.type().device_type() == at::kCUDA, #x " must be on CPU or CUDA")

// 2x2 matrix [a, b; c, d] multiplied by a vector [x, y]
template <typename scalar_t>
static inline std::pair<scalar_t, scalar_t> mult2x2(scalar_t a, scalar_t b,
                                                    scalar_t c, scalar_t d,
                                                    scalar_t x, scalar_t y) {
  return std::make_pair(a * x + b * y, c * x + d * y);
}

at::Tensor butterfly_factor_multiply(const at::Tensor& twiddle, const at::Tensor& input) {
  /* Parameters:
        twiddle: (2, 2, n) if real or (2, 2, n, 2) if complex
        input: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
     Return:
        output: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
  */
  auto output = torch::empty_like(input);
  if (input.is_cuda()) {
    AT_CHECK(twiddle.is_cuda(), "butterfly_factor_multiply: Expected twiddle to be CUDA tensor");
    butterfly_factor_multiply_cuda(twiddle, input, output);
    return output;
  }
  AT_CHECK(!twiddle.is_cuda(), "butterfly_factor_multiply: Expected twiddle to be CPU tensor");
  const auto batch_size = input.size(0);
  const auto n = input.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_factor_multiply", [&] {
    switch (input.dim()) {
      case 3:  // real
        {
          const auto twiddle_a = twiddle.accessor<scalar_t, 3>();
          const auto input_a = input.accessor<scalar_t, 3>();
          auto output_a = output.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n; ++i) {
              const scalar_t twiddle_val[2][2] = {{twiddle_a[0][0][i], twiddle_a[0][1][i]},
                                                  {twiddle_a[1][0][i], twiddle_a[1][1][i]}};
              const scalar_t input_val[2] = {input_a[b][0][i], input_a[b][1][i]};
              for (int64_t j = 0; j <= 1; ++j) {
                output_a[b][j][i] = twiddle_val[j][0] * input_val[0] + twiddle_val[j][1] * input_val[1];
              }
            }
          }
          break;
        }
      case 4:  // complex
        {
          const auto twiddle_a = twiddle.accessor<scalar_t, 4>();
          const auto input_a = input.accessor<scalar_t, 4>();
          auto output_a = output.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n; ++i) {
              const scalar_t twiddle_val[2][2][2] = {{{twiddle_a[0][0][i][0], twiddle_a[0][0][i][1]},
                                                      {twiddle_a[0][1][i][0], twiddle_a[0][1][i][1]}},
                                                     {{twiddle_a[1][0][i][0], twiddle_a[1][0][i][1]},
                                                      {twiddle_a[1][1][i][0], twiddle_a[1][1][i][1]}}};
              const scalar_t input_val[2][2] = {{input_a[b][0][i][0], input_a[b][0][i][1]},
                                                {input_a[b][1][i][0], input_a[b][1][i][1]}};
              for (int64_t j = 0; j <= 1; ++j) {
                output_a[b][j][i][0] = twiddle_val[j][0][0] * input_val[0][0] - twiddle_val[j][0][1] * input_val[0][1]
                  + twiddle_val[j][1][0] * input_val[1][0] - twiddle_val[j][1][1] * input_val[1][1];
                output_a[b][j][i][1] = twiddle_val[j][0][0] * input_val[0][1] + twiddle_val[j][0][1] * input_val[0][0]
                  + twiddle_val[j][1][0] * input_val[1][1] + twiddle_val[j][1][1] * input_val[1][0];
              }
            }
          }
          break;
        }
      default:
        AT_ERROR("butterfly_factor_multiply requires input dimension 3 or 4");
    }
  });
  return output;
}

std::vector<at::Tensor> butterfly_factor_multiply_backward(const at::Tensor& grad, const at::Tensor& twiddle, const at::Tensor& input) {
  /* Parameters:
         grad: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
         twiddle: (2, 2, n) if real or (2, 2, n, 2) if complex
         input: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
     Return:
         d_twiddle: (2, 2, n) if real or (2, 2, n, 2) if complex
         d_input: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
  */
  const auto batch_size = input.size(0);
  const auto n = input.size(2);
  auto d_input = torch::empty_like(input);
  if (input.is_cuda()) {
    AT_CHECK(twiddle.is_cuda() && grad.is_cuda(), "butterfly_factor_multiply_backward: Expected grad and twiddle to be CUDA tensor");
    // CUDA kernel will compute the expanded gradient of @twiddle, then we'll call sum over the batch dimension.
    // This is because I haven't figured out how to write efficient reduction kernel in CUDA.
    auto d_twiddle_expanded = input.dim() == 3 ?
      // torch::empty({(batch_size + 3) / 4, 2, 2, n}, torch::dtype(twiddle.dtype()).device(twiddle.device())) :
      // torch::empty({batch_size, 2, 2, n}, torch::dtype(twiddle.dtype()).device(twiddle.device())) :
      torch::zeros({2, 2, n}, torch::dtype(twiddle.dtype()).device(twiddle.device())) :
      torch::empty({batch_size, 2, 2, n, 2}, torch::dtype(twiddle.dtype()).device(twiddle.device()));
    butterfly_factor_multiply_backward_cuda(grad, twiddle, input, d_twiddle_expanded, d_input);
    // return {d_twiddle_expanded.sum(0), d_input};
    // return {d_twiddle_expanded[0], d_input};
    return {d_twiddle_expanded, d_input};
  }
  AT_CHECK((!twiddle.is_cuda()) && (!grad.is_cuda()) , "butterfly_factor_multiply_backward: Expected grad and twiddle to be CPU tensor");
  auto d_twiddle = torch::zeros_like(twiddle);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_factor_multiply_backward", [&] {
    switch (input.dim()) {
      case 3:  // real
        {
          const auto grad_a = grad.accessor<scalar_t, 3>();
          const auto twiddle_a = twiddle.accessor<scalar_t, 3>();
          const auto input_a = input.accessor<scalar_t, 3>();
          auto d_twiddle_a = d_twiddle.accessor<scalar_t, 3>();
          auto d_input_a = d_input.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n; ++i) {
              const scalar_t twiddle_val[2][2] = {{twiddle_a[0][0][i], twiddle_a[0][1][i]},
                                                  {twiddle_a[1][0][i], twiddle_a[1][1][i]}};
              const scalar_t input_val[2] = {input_a[b][0][i], input_a[b][1][i]};
              const scalar_t grad_val[2] = {grad_a[b][0][i], grad_a[b][1][i]};
              for (int64_t j = 0; j <= 1; ++j) {
                d_twiddle_a[j][0][i] += grad_val[j] * input_val[0];
                d_twiddle_a[j][1][i] += grad_val[j] * input_val[1];
                d_input_a[b][j][i] = twiddle_val[0][j] * grad_val[0] + twiddle_val[1][j] * grad_val[1];
              }
            }
          }
          break;
        }
      case 4:  // complex
        {
          const auto grad_a = grad.accessor<scalar_t, 4>();
          const auto twiddle_a = twiddle.accessor<scalar_t, 4>();
          const auto input_a = input.accessor<scalar_t, 4>();
          auto d_twiddle_a = d_twiddle.accessor<scalar_t, 4>();
          auto d_input_a = d_input.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n; ++i) {
              const scalar_t twiddle_val[2][2][2] = {{{twiddle_a[0][0][i][0], twiddle_a[0][0][i][1]},
                                                      {twiddle_a[0][1][i][0], twiddle_a[0][1][i][1]}},
                                                     {{twiddle_a[1][0][i][0], twiddle_a[1][0][i][1]},
                                                      {twiddle_a[1][1][i][0], twiddle_a[1][1][i][1]}}};
              const scalar_t input_val[2][2] = {{input_a[b][0][i][0], input_a[b][0][i][1]},
                                                {input_a[b][1][i][0], input_a[b][1][i][1]}};
              const scalar_t grad_val[2][2] = {{grad_a[b][0][i][0], grad_a[b][0][i][1]},
                                               {grad_a[b][1][i][0], grad_a[b][1][i][1]}};
              for (int64_t j = 0; j <= 1; ++j) {
                // Multiply by complex conjugate
                d_twiddle_a[j][0][i][0] += grad_val[j][0] * input_val[0][0] + grad_val[j][1] * input_val[0][1];
                d_twiddle_a[j][0][i][1] += -grad_val[j][0] * input_val[0][1] + grad_val[j][1] * input_val[0][0];
                d_twiddle_a[j][1][i][0] += grad_val[j][0] * input_val[1][0] + grad_val[j][1] * input_val[1][1];
                d_twiddle_a[j][1][i][1] += -grad_val[j][0] * input_val[1][1] + grad_val[j][1] * input_val[1][0];
                d_input_a[b][j][i][0] = twiddle_val[0][j][0] * grad_val[0][0] + twiddle_val[0][j][1] * grad_val[0][1]
                  + twiddle_val[1][j][0] * grad_val[1][0] + twiddle_val[1][j][1] * grad_val[1][1];
                d_input_a[b][j][i][1] = twiddle_val[0][j][0] * grad_val[0][1] - twiddle_val[0][j][1] * grad_val[0][0]
                  + twiddle_val[1][j][0] * grad_val[1][1] - twiddle_val[1][j][1] * grad_val[1][0];
              }
            }
          }
          break;
        }
      default:
        AT_ERROR("butterfly_factor_multiply_backward requires input dimension 3 or 4");
    }
  });
  return {d_twiddle, d_input};
}

at::Tensor butterfly_multiply_inplace(const at::Tensor& twiddle, const at::Tensor& input) {
  /* Parameters:
         twiddle: (n - 1, 2, 2) if real or (n - 1, 2, 2, 2) if complex
         input: (batch_size, n) if real or (batch_size, n, 2) if complex
     Returns:
         output: (batch_size, n) if real or (batch_size, n, 2) if complex
  */
  const auto batch_size = input.size(0);
  const auto n = input.size(1);
  auto output = input.clone();
  if (output.is_cuda()) {
    AT_CHECK(twiddle.is_cuda(), "butterfly_multiply_inplace: Expected twiddle to be CUDA tensor");
    // butterfly_multiply_inplace_cuda(twiddle, output);
    // int m = int(log2((double) input.size(1)));
    auto input_temp_phi = input.dim() == 3 ?
      torch::empty({batch_size, n, 2}, torch::dtype(twiddle.dtype()).device(twiddle.device())) :
      torch::empty({batch_size, n}, torch::dtype(twiddle.dtype()).device(twiddle.device()));
    butterfly_multiply_inplace_cuda(twiddle, output);
    return output;
  }
  AT_CHECK(!twiddle.is_cuda(), "butterfly_multiply_inplace: Expected twiddle to be CPU tensor");
  // const auto batch_size = output.size(0);
  // const auto n = output.size(1);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.type(), "butterfly_multiply_inplace", [&] {
    switch (output.dim()) {
      case 2:  // real
        {
          const auto twiddle_a = twiddle.accessor<scalar_t, 3>();
          auto output_a = output.accessor<scalar_t, 2>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t stride = 1; stride <= n / 2; stride *= 2) {
              int64_t twiddle_start_idx = stride - 1;
              for (int64_t i = 0; i < n / 2; ++i) {
                int64_t low_order_bit = i % stride;
                int64_t twiddle_idx = twiddle_start_idx + low_order_bit;
                int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
                const scalar_t twiddle_val[2][2] = {{twiddle_a[twiddle_idx][0][0], twiddle_a[twiddle_idx][0][1]},
                                                    {twiddle_a[twiddle_idx][1][0], twiddle_a[twiddle_idx][1][1]}};
                const scalar_t input_val[2] = {output_a[b][pos], output_a[b][pos + stride]};
                output_a[b][pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
                output_a[b][pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
              }
            }
          }
          break;
        }
      case 3:  // complex
        {
          const auto twiddle_a = twiddle.accessor<scalar_t, 4>();
          auto output_a = output.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t stride = 1; stride <= n / 2; stride *= 2) {
              int64_t twiddle_start_idx = stride - 1;
              for (int64_t i = 0; i < n / 2; ++i) {
                int64_t low_order_bit = i % stride;
                int64_t twiddle_idx = twiddle_start_idx + low_order_bit;
                int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
                const scalar_t twiddle_val[2][2][2] = {{{twiddle_a[twiddle_idx][0][0][0], twiddle_a[twiddle_idx][0][0][1]},
                                                        {twiddle_a[twiddle_idx][0][1][0], twiddle_a[twiddle_idx][0][1][1]}},
                                                       {{twiddle_a[twiddle_idx][1][0][0], twiddle_a[twiddle_idx][1][0][1]},
                                                        {twiddle_a[twiddle_idx][1][1][0], twiddle_a[twiddle_idx][1][1][1]}}};
                const scalar_t input_val[2][2] = {{output_a[b][pos][0], output_a[b][pos][1]},
                                                  {output_a[b][pos + stride][0], output_a[b][pos + stride][1]}};
                output_a[b][pos][0] = twiddle_val[0][0][0] * input_val[0][0] - twiddle_val[0][0][1] * input_val[0][1]
                  + twiddle_val[0][1][0] * input_val[1][0] - twiddle_val[0][1][1] * input_val[1][1];
                output_a[b][pos][1] = twiddle_val[0][0][0] * input_val[0][1] + twiddle_val[0][0][1] * input_val[0][0]
                  + twiddle_val[0][1][0] * input_val[1][1] + twiddle_val[0][1][1] * input_val[1][0];
                output_a[b][pos + stride][0] = twiddle_val[1][0][0] * input_val[0][0] - twiddle_val[1][0][1] * input_val[0][1]
                  + twiddle_val[1][1][0] * input_val[1][0] - twiddle_val[1][1][1] * input_val[1][1];
                output_a[b][pos + stride][1] = twiddle_val[1][0][0] * input_val[0][1] + twiddle_val[1][0][1] * input_val[0][0]
                  + twiddle_val[1][1][0] * input_val[1][1] + twiddle_val[1][1][1] * input_val[1][0];
              }
              twiddle_start_idx += stride;
            }
          }
          break;
        }
      default:
        AT_ERROR("butterfly_multiply_inplace requires input dimension 2 or 3");
    }
  });
  return output;
}

std::vector<at::Tensor> butterfly_multiply_inplace_backward(const at::Tensor& grad, const at::Tensor& twiddle, const at::Tensor& output) {
  /* Parameters:
         grad: (batch_size, n) if real or (batch_size, n, 2) if complex
         twiddle: (n - 1, 2, 2) if real or (n - 1, 2, 2, 2) if complex
         output: (batch_size, n) if real or (batch_size, n, 2) if complex
     Return:
         d_twiddle: (n - 1, 2, 2) if real or (n - 1, 2, 2, 2) if complex
         d_input: (batch_size, n) if real or (batch_size, n, 2) if complex
  */
  const auto batch_size = output.size(0);
  const auto n = output.size(1);
  auto d_input = grad.clone();
  auto d_twiddle = torch::zeros_like(twiddle);
  auto output_clone = at::_cast_Double(output.clone());
  if (output.is_cuda()) {
    AT_CHECK(twiddle.is_cuda() && grad.is_cuda(), "butterfly_multiply_inplace_backward: Expected grad and twiddle to be CUDA tensor");
    butterfly_multiply_inplace_backward_cuda(grad, twiddle, output_clone, d_twiddle, d_input);
    return {d_twiddle, d_input};
  }
  AT_CHECK((!twiddle.is_cuda()) && (!grad.is_cuda()) , "butterfly_multiply_inplace_backward: Expected grad and twiddle to be CPU tensor");
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "butterfly_multiply_inplace_backward", [&] {
    switch (grad.dim()) {
      case 2:  // real
        {
          const auto twiddle_a = twiddle.accessor<scalar_t, 3>();
          // auto output_a = output_clone.accessor<scalar_t, 2>();
          auto output_a = output_clone.accessor<double, 2>();
          auto d_twiddle_a = d_twiddle.accessor<scalar_t, 3>();
          auto d_input_a = d_input.accessor<scalar_t, 2>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t stride = n / 2; stride >= 1; stride /= 2) {
              int64_t twiddle_start_idx = stride - 1;
              for (int64_t i = 0; i < n / 2; ++i) {
                int64_t low_order_bit = i % stride;
                int64_t twiddle_idx = twiddle_start_idx + low_order_bit;
                int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
                const scalar_t twiddle_val[2][2] = {{twiddle_a[twiddle_idx][0][0], twiddle_a[twiddle_idx][0][1]},
                                                    {twiddle_a[twiddle_idx][1][0], twiddle_a[twiddle_idx][1][1]}};
                const scalar_t grad_val[2] = {d_input_a[b][pos], d_input_a[b][pos + stride]};
                d_input_a[b][pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
                d_input_a[b][pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
                // const scalar_t output_val[2] = {output_a[b][pos], output_a[b][pos + stride]};
                const double output_val[2] = {output_a[b][pos], output_a[b][pos + stride]};
                // const scalar_t twiddle_det_inv = 1.0 / (twiddle_val[0][0] * twiddle_val[1][1] - twiddle_val[0][1] * twiddle_val[1][0]);
                const double twiddle_det_inv = 1.0 / (twiddle_val[0][0] * twiddle_val[1][1] - twiddle_val[0][1] * twiddle_val[1][0]);
                // const scalar_t input_val[2] = {(twiddle_val[1][1] * output_val[0] - twiddle_val[0][1] * output_val[1]) * twiddle_det_inv,
                const double input_val[2] = {(twiddle_val[1][1] * output_val[0] - twiddle_val[0][1] * output_val[1]) * twiddle_det_inv,
                                               (-twiddle_val[1][0] * output_val[0] + twiddle_val[0][0] * output_val[1]) * twiddle_det_inv};
                output_a[b][pos] = input_val[0];
                output_a[b][pos + stride] = input_val[1];
                d_twiddle_a[twiddle_idx][0][0] += grad_val[0] * input_val[0];
                d_twiddle_a[twiddle_idx][0][1] += grad_val[0] * input_val[1];
                d_twiddle_a[twiddle_idx][1][0] += grad_val[1] * input_val[0];
                d_twiddle_a[twiddle_idx][1][1] += grad_val[1] * input_val[1];
              }
            }
          }
          break;
        }
      case 3:  // complex
        {
          // const auto twiddle_a = twiddle.accessor<scalar_t, 4>();
          // const auto output_a = output.accessor<scalar_t, 3>();
          // auto d_twiddle_a = d_twiddle.accessor<scalar_t, 4>();
          // auto d_input_a = d_input.accessor<scalar_t, 3>();
          // for (int64_t b = 0; b < batch_size; ++b) {
          //   for (int64_t i = 0; i < n / 2; ++i) {
          //     const scalar_t twiddle_val[2][2][2] = {{{twiddle_a[0][0][i][0], twiddle_a[0][0][i][1]},
          //                                             {twiddle_a[0][1][i][0], twiddle_a[0][1][i][1]}},
          //                                            {{twiddle_a[1][0][i][0], twiddle_a[1][0][i][1]},
          //                                             {twiddle_a[1][1][i][0], twiddle_a[1][1][i][1]}}};
          //     const scalar_t input_val[2][2] = {{input_a[b][0][i][0], input_a[b][0][i][1]},
          //                                       {input_a[b][1][i][0], input_a[b][1][i][1]}};
          //     const scalar_t grad_val[2][2] = {{grad_a[b][0][i][0], grad_a[b][0][i][1]},
          //                                      {grad_a[b][1][i][0], grad_a[b][1][i][1]}};
          //     for (int64_t j = 0; j <= 1; ++j) {
          //       // Multiply by complex conjugate
          //       d_twiddle_a[j][0][i][0] += grad_val[j][0] * input_val[0][0] + grad_val[j][1] * input_val[0][1];
          //       d_twiddle_a[j][0][i][1] += -grad_val[j][0] * input_val[0][1] + grad_val[j][1] * input_val[0][0];
          //       d_twiddle_a[j][1][i][0] += grad_val[j][0] * input_val[1][0] + grad_val[j][1] * input_val[1][1];
          //       d_twiddle_a[j][1][i][1] += -grad_val[j][0] * input_val[1][1] + grad_val[j][1] * input_val[1][0];
          //       d_input_a[b][j][i][0] = twiddle_val[0][j][0] * grad_val[0][0] + twiddle_val[0][j][1] * grad_val[0][1]
          //         + twiddle_val[1][j][0] * grad_val[1][0] + twiddle_val[1][j][1] * grad_val[1][1];
          //       d_input_a[b][j][i][1] = twiddle_val[0][j][0] * grad_val[0][1] - twiddle_val[0][j][1] * grad_val[0][0]
          //         + twiddle_val[1][j][0] * grad_val[1][1] - twiddle_val[1][j][1] * grad_val[1][0];
          //     }
          //   }
          // }
          break;
        }
      default:
        AT_ERROR("butterfly_multiply_inplace_backward requires input dimension 2 or 3");
    }
  });
  return {d_twiddle, d_input};
}

at::Tensor butterfly_multiply_intermediate(const at::Tensor& twiddle, const at::Tensor& input, bool increasing_stride, bool return_intermediates) {
  /* Parameters:
         twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
         input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
         return_intermediates: whether to return just the output (i.e. computed in-place) or output
             and intermediate values for backward pass.
     Returns:
         if return_intermediates:
             output + intermediate values for backward pass: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
         else:
             output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
  */
  const auto batch_size = input.size(0);
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_CHECK((twiddle.dim() == 4 && input.dim() == 3) || (twiddle.dim() == 5 && input.dim() == 4),
           "butterfly_multiply_intermediate: twiddle and input must have dimension 4,3 or 5,4");
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(input);
  AT_CHECK(twiddle.device() == input.device(), "device of twiddle (", twiddle.device(), ") must match device of input (", input.device(), ")");
  AT_CHECK(twiddle.size(0) == nstack && twiddle.size(1) == n - 1 && twiddle.size(2) == 2 && twiddle.size(3) == 2, "butterfly_multiply_intermediate: twiddle must have shape (nstack, n-1, 2, 2) or (nstack, n-1, 2, 2, 2)");
  const int output_first_dim = return_intermediates ? log_n + 1 : 1;
  auto output = input.dim() == 3 ?
    torch::empty({output_first_dim, batch_size, nstack, n}, torch::dtype(input.dtype()).device(input.device())) :
    torch::empty({output_first_dim, batch_size, nstack, n, 2}, torch::dtype(input.dtype()).device(input.device()));
  if (!return_intermediates) {
    output = input.dim() == 3 ? output.expand({log_n + 1, batch_size, nstack, n})
                              : output.expand({log_n + 1, batch_size, nstack, n, 2});
  }
  output[0] = input;
  if (input.is_cuda()) {
    butterfly_multiply_intermediate_cuda(twiddle, output, increasing_stride, return_intermediates);
    return return_intermediates ? output : output[-1];
  }
  const bool complex = input.dim() == 4;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_multiply_intermediate", [&] {
    if (!complex) {  // real
      const auto twiddle_a = twiddle.accessor<scalar_t, 4>();
      auto output_a = output.accessor<scalar_t, 4>();
      for (int64_t idx = 0; idx <= log_n - 1; ++idx) {
        int64_t log_stride = increasing_stride ? idx : (log_n - 1 - idx);
        int64_t stride = 1 << log_stride;
        int64_t twiddle_start_idx = stride - 1;
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t s = 0; s < nstack; ++s) {
            for (int64_t i = 0; i < n / 2; ++i) {
              int64_t low_order_bit = i % stride;
              int64_t twiddle_idx = twiddle_start_idx + low_order_bit;
              int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
              const scalar_t twiddle_val[2][2] = {{twiddle_a[s][twiddle_idx][0][0], twiddle_a[s][twiddle_idx][0][1]},
                                                  {twiddle_a[s][twiddle_idx][1][0], twiddle_a[s][twiddle_idx][1][1]}};
              const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
              output_a[idx+1][b][s][pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
              output_a[idx+1][b][s][pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
            }
          }
        }
      }
    } else {  // complex
      using complex_t = std::complex<scalar_t>;
      const auto twiddle_a = twiddle.accessor<scalar_t, 5>();
      auto output_a = output.accessor<scalar_t, 5>();
      for (int64_t idx = 0; idx <= log_n - 1; ++idx) {
        int64_t log_stride = increasing_stride ? idx : log_n - 1 - idx;
        int64_t stride = 1 << log_stride;
        int64_t twiddle_start_idx = stride - 1;
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t s = 0; s < nstack; ++s) {
            for (int64_t i = 0; i < n / 2; ++i) {
              int64_t low_order_bit = i % stride;
              int64_t twiddle_idx = twiddle_start_idx + low_order_bit;
              int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
              const complex_t twiddle_val[2][2] =
                {{complex_t(twiddle_a[s][twiddle_idx][0][0][0], twiddle_a[s][twiddle_idx][0][0][1]),
                  complex_t(twiddle_a[s][twiddle_idx][0][1][0], twiddle_a[s][twiddle_idx][0][1][1])},
                 {complex_t(twiddle_a[s][twiddle_idx][1][0][0], twiddle_a[s][twiddle_idx][1][0][1]),
                  complex_t(twiddle_a[s][twiddle_idx][1][1][0], twiddle_a[s][twiddle_idx][1][1][1])}};
              const complex_t input_val[2] =
                {complex_t(output_a[idx][b][s][pos][0], output_a[idx][b][s][pos][1]),
                 complex_t(output_a[idx][b][s][pos + stride][0], output_a[idx][b][s][pos + stride][1])};
              const complex_t output_val[2] =
                {twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1],
                twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1]};
              // output_a[idx+1][b][s][pos][0] = std::real(output_val[0]);
              output_a[idx+1][b][s][pos][0] = output_val[0].real();
              output_a[idx+1][b][s][pos][1] = output_val[0].imag();
              output_a[idx+1][b][s][pos + stride][0] = output_val[1].real();
              output_a[idx+1][b][s][pos + stride][1] = output_val[1].imag();
            }
          }
        }
      }
    }
  });
  return return_intermediates ? output : output[-1];
}

std::vector<at::Tensor> butterfly_multiply_intermediate_backward(const at::Tensor& grad, const at::Tensor& twiddle, const at::Tensor& output, bool increasing_stride) {
  /* Parameters:
         grad: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
         twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
         output + intermediate values for backward: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
         increasing_stride: whether the forward pass multiply was with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
     Return:
         d_twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
         d_input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
  */
  const auto batch_size = grad.size(0);
  const auto nstack = grad.size(1);
  const auto n = grad.size(2);
  const int log_n = int(log2((double) n));
  AT_CHECK((grad.dim() == 3 && twiddle.dim() == 4 && output.dim() == 4) || (grad.dim() == 4 && twiddle.dim() == 5 && output.dim() == 5),
           "butterfly_multiply_intermediate_backward: grad, twiddle, and output must have dimension 3,4,4 or 4,5,5");
  CHECK_DEVICE(grad);
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(output);
  AT_CHECK(grad.device() == twiddle.device() && twiddle.device() == output.device(), "device of grad (", grad.device(), ")twiddle (", twiddle.device(), "), and output (", output.device(), ") must match");
  AT_CHECK(twiddle.size(0) == nstack && twiddle.size(1) == n - 1 && twiddle.size(2) == 2 && twiddle.size(3) == 2, "butterfly_multiply_intermediate_backward: twiddle must have shape (nstack, n-1, 2, 2) or (nstack, n-1, 2, 2, 2)");
  AT_CHECK(output.size(0) == log_n + 1 && output.size(1) == batch_size && output.size(2) == nstack && output.size(3) == n, "butterfly_multiply_intermediate_backward: output must have shape (log n + 1, batch_size, nstack, n) or (log n + 1, batch_size, nstack, n, 2)");
  auto d_input = grad.clone();
  auto d_twiddle = torch::zeros_like(twiddle);
  if (output.is_cuda()) {
    butterfly_multiply_intermediate_backward_cuda(twiddle, output, d_twiddle, d_input, increasing_stride);
    return {d_twiddle, d_input} ;
  }
  bool complex = grad.dim() == 4;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "butterfly_multiply_intermediate_backward", [&] {
    if (!complex) {
      const auto twiddle_a = twiddle.accessor<scalar_t, 4>();
      auto output_a = output.accessor<scalar_t, 4>();
      auto d_twiddle_a = d_twiddle.accessor<scalar_t, 4>();
      auto d_input_a = d_input.accessor<scalar_t, 3>();
      for (int64_t idx = log_n - 1; idx >= 0; --idx) {
        int64_t log_stride = increasing_stride ? idx : log_n - 1 - idx;
        int64_t stride = 1 << log_stride;
        int64_t twiddle_start_idx = stride - 1;
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t s = 0; s < nstack; ++s) {
            for (int64_t i = 0; i < n / 2; ++i) {
              int64_t low_order_bit = i % stride;
              int64_t twiddle_idx = twiddle_start_idx + low_order_bit;
              int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
              const scalar_t twiddle_val[2][2] = {{twiddle_a[s][twiddle_idx][0][0], twiddle_a[s][twiddle_idx][0][1]},
                                                  {twiddle_a[s][twiddle_idx][1][0], twiddle_a[s][twiddle_idx][1][1]}};
              const scalar_t grad_val[2] = {d_input_a[b][s][pos], d_input_a[b][s][pos + stride]};
              d_input_a[b][s][pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
              d_input_a[b][s][pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
              const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
              d_twiddle_a[s][twiddle_idx][0][0] += grad_val[0] * input_val[0];
              d_twiddle_a[s][twiddle_idx][0][1] += grad_val[0] * input_val[1];
              d_twiddle_a[s][twiddle_idx][1][0] += grad_val[1] * input_val[0];
              d_twiddle_a[s][twiddle_idx][1][1] += grad_val[1] * input_val[1];
            }
          }
        }
      }
    } else {  // complex
      using complex_t = std::complex<scalar_t>;
      const auto twiddle_a = twiddle.accessor<scalar_t, 5>();
      const auto output_a = output.accessor<scalar_t, 5>();
      auto d_twiddle_a = d_twiddle.accessor<scalar_t, 5>();
      auto d_input_a = d_input.accessor<scalar_t, 4>();
      for (int64_t idx = log_n - 1; idx >= 0; --idx) {
        int64_t log_stride = increasing_stride ? idx : log_n - 1 - idx;
        int64_t stride = 1 << log_stride;
        int64_t twiddle_start_idx = stride - 1;
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t s = 0; s < nstack; ++s) {
            for (int64_t i = 0; i < n / 2; ++i) {
              int64_t low_order_bit = i % stride;
              int64_t twiddle_idx = twiddle_start_idx + low_order_bit;
              int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
              const complex_t twiddle_val[2][2] =
                {{complex_t(twiddle_a[s][twiddle_idx][0][0][0], twiddle_a[s][twiddle_idx][0][0][1]),
                  complex_t(twiddle_a[s][twiddle_idx][0][1][0], twiddle_a[s][twiddle_idx][0][1][1])},
                 {complex_t(twiddle_a[s][twiddle_idx][1][0][0], twiddle_a[s][twiddle_idx][1][0][1]),
                  complex_t(twiddle_a[s][twiddle_idx][1][1][0], twiddle_a[s][twiddle_idx][1][1][1])}};
              const complex_t grad_val[2] =
                {complex_t(d_input_a[b][s][pos][0], d_input_a[b][s][pos][1]),
                 complex_t(d_input_a[b][s][pos + stride][0], d_input_a[b][s][pos + stride][1])};
              const complex_t d_input_val[2] =
                {std::conj(twiddle_val[0][0]) * grad_val[0] + std::conj(twiddle_val[1][0]) * grad_val[1],
                 std::conj(twiddle_val[0][1]) * grad_val[0] + std::conj(twiddle_val[1][1]) * grad_val[1]};
              d_input_a[b][s][pos][0] = d_input_val[0].real();
              d_input_a[b][s][pos][1] = d_input_val[0].imag();
              d_input_a[b][s][pos + stride][0] = d_input_val[1].real();
              d_input_a[b][s][pos + stride][1] = d_input_val[1].imag();
              const complex_t input_val[2] =
                {complex_t(output_a[idx][b][s][pos][0], output_a[idx][b][s][pos][1]),
                 complex_t(output_a[idx][b][s][pos + stride][0], output_a[idx][b][s][pos + stride][1])};
              const complex_t d_twiddle_val[2][2] =
                {{grad_val[0] * std::conj(input_val[0]), grad_val[0] * std::conj(input_val[1])},
                 {grad_val[1] * std::conj(input_val[0]), grad_val[1] * std::conj(input_val[1])}};
              d_twiddle_a[s][twiddle_idx][0][0][0] += d_twiddle_val[0][0].real();
              d_twiddle_a[s][twiddle_idx][0][0][1] += d_twiddle_val[0][0].imag();
              d_twiddle_a[s][twiddle_idx][0][1][0] += d_twiddle_val[0][1].real();
              d_twiddle_a[s][twiddle_idx][0][1][1] += d_twiddle_val[0][1].imag();
              d_twiddle_a[s][twiddle_idx][1][0][0] += d_twiddle_val[1][0].real();
              d_twiddle_a[s][twiddle_idx][1][0][1] += d_twiddle_val[1][0].imag();
              d_twiddle_a[s][twiddle_idx][1][1][0] += d_twiddle_val[1][1].real();
              d_twiddle_a[s][twiddle_idx][1][1][1] += d_twiddle_val[1][1].imag();
            }
          }
        }
      }
    }
  });
  return {d_twiddle, d_input} ;
}

at::Tensor butterfly_multiply_untied(const at::Tensor& twiddle, const at::Tensor& input, bool increasing_stride, bool return_intermediates) {
  /* Parameters:
         twiddle: (nstack, log n, n/2, 2, 2) if real or (nstack, log n, n/2, 2, 2, 2) if complex
         input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
         return_intermediates: whether to return just the output (i.e. computed in-place) or output
             and intermediate values for backward pass.
     Returns:
         if return_intermediates:
             output + intermediate values for backward pass: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
         else:
             output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
  */
  const auto batch_size = input.size(0);
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_CHECK((twiddle.dim() == 5 && input.dim() == 3) || (twiddle.dim() == 6 && input.dim() == 4),
           "butterfly_multiply_untied: twiddle and input must have dimension 5,3 or 6,4");
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(input);
  AT_CHECK(twiddle.device() == input.device(), "device of twiddle (", twiddle.device(), ") must match device of input (", input.device(), ")");
  AT_CHECK(twiddle.size(0) == nstack && twiddle.size(1) == log_n && twiddle.size(2) == n / 2 && twiddle.size(3) == 2 && twiddle.size(4) == 2, "butterfly_multiply_untied: twiddle must have shape (nstack, log n, n/2, 2, 2) or (nstack, log n, n/2, 2, 2, 2)");
  const int output_first_dim = return_intermediates ? log_n + 1 : 1;
  auto output = input.dim() == 3 ?
    torch::empty({output_first_dim, batch_size, nstack, n}, torch::dtype(input.dtype()).device(input.device())) :
    torch::empty({output_first_dim, batch_size, nstack, n, 2}, torch::dtype(input.dtype()).device(input.device()));
  if (!return_intermediates) {
    output = input.dim() == 3 ? output.expand({log_n + 1, batch_size, nstack, n})
                              : output.expand({log_n + 1, batch_size, nstack, n, 2});
  }
  output[0] = input;
  if (input.is_cuda()) {
    butterfly_multiply_untied_cuda(twiddle, output, increasing_stride, return_intermediates);
    return return_intermediates ? output : output[-1];
  }
  const bool complex = input.dim() == 4;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_multiply_untied", [&] {
    if (!complex) {  // real
      const auto twiddle_a = twiddle.accessor<scalar_t, 5>();
      auto output_a = output.accessor<scalar_t, 4>();
      for (int64_t idx = 0; idx <= log_n - 1; ++idx) {
        int64_t log_stride = increasing_stride ? idx : (log_n - 1 - idx);
        int64_t stride = 1 << log_stride;
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t s = 0; s < nstack; ++s) {
            for (int64_t i = 0; i < n / 2; ++i) {
              int64_t low_order_bit = i % stride;
              int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
              const scalar_t twiddle_val[2][2] = {{twiddle_a[s][log_stride][i][0][0], twiddle_a[s][log_stride][i][0][1]},
                                                  {twiddle_a[s][log_stride][i][1][0], twiddle_a[s][log_stride][i][1][1]}};
              const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
              output_a[idx+1][b][s][pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
              output_a[idx+1][b][s][pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
            }
          }
        }
      }
    } else {  // complex
      using complex_t = std::complex<scalar_t>;
      const auto twiddle_a = twiddle.accessor<scalar_t, 6>();
      auto output_a = output.accessor<scalar_t, 5>();
      for (int64_t idx = 0; idx <= log_n - 1; ++idx) {
        int64_t log_stride = increasing_stride ? idx : log_n - 1 - idx;
        int64_t stride = 1 << log_stride;
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t s = 0; s < nstack; ++s) {
            for (int64_t i = 0; i < n / 2; ++i) {
              int64_t low_order_bit = i % stride;
              int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
              const complex_t twiddle_val[2][2] =
                {{complex_t(twiddle_a[s][log_stride][i][0][0][0], twiddle_a[s][log_stride][i][0][0][1]),
                  complex_t(twiddle_a[s][log_stride][i][0][1][0], twiddle_a[s][log_stride][i][0][1][1])},
                 {complex_t(twiddle_a[s][log_stride][i][1][0][0], twiddle_a[s][log_stride][i][1][0][1]),
                  complex_t(twiddle_a[s][log_stride][i][1][1][0], twiddle_a[s][log_stride][i][1][1][1])}};
              const complex_t input_val[2] =
                {complex_t(output_a[idx][b][s][pos][0], output_a[idx][b][s][pos][1]),
                 complex_t(output_a[idx][b][s][pos + stride][0], output_a[idx][b][s][pos + stride][1])};
              const complex_t output_val[2] =
                {twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1],
                 twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1]};
              // output_a[idx+1][b][s][pos][0] = std::real(output_val[0]);
              output_a[idx+1][b][s][pos][0] = output_val[0].real();
              output_a[idx+1][b][s][pos][1] = output_val[0].imag();
              output_a[idx+1][b][s][pos + stride][0] = output_val[1].real();
              output_a[idx+1][b][s][pos + stride][1] = output_val[1].imag();
            }
          }
        }
      }
    }
  });
  return return_intermediates ? output : output[-1];
}

std::vector<at::Tensor> butterfly_multiply_untied_backward(const at::Tensor& grad, const at::Tensor& twiddle, const at::Tensor& output, bool increasing_stride) {
  /* Parameters:
         grad: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
         twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
         output + untied values for backward: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
         increasing_stride: whether the forward pass multiply was with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
     Return:
         d_twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
         d_input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
  */
  const auto batch_size = grad.size(0);
  const auto nstack = grad.size(1);
  const auto n = grad.size(2);
  const int log_n = int(log2((double) n));
  AT_CHECK((grad.dim() == 3 && twiddle.dim() == 5 && output.dim() == 4) || (grad.dim() == 4 && twiddle.dim() == 6 && output.dim() == 5),
           "butterfly_multiply_untied_backward: grad, twiddle, and output must have dimension 3,5,4 or 4,6,5");
  CHECK_DEVICE(grad);
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(output);
  AT_CHECK(grad.device() == twiddle.device() && twiddle.device() == output.device(), "device of grad (", grad.device(), ")twiddle (", twiddle.device(), "), and output (", output.device(), ") must match");
  AT_CHECK(twiddle.size(0) == nstack && twiddle.size(1) == log_n && twiddle.size(2) == n / 2 && twiddle.size(3) == 2 && twiddle.size(4) == 2, "butterfly_multiply_untied_backward: twiddle must have shape (nstack, log n, n/2, 2, 2) or (nstack, log n, n/2, 2, 2, 2)");
  AT_CHECK(output.size(0) == log_n + 1 && output.size(1) == batch_size && output.size(2) == nstack && output.size(3) == n, "butterfly_multiply_untied_backward: output must have shape (log n + 1, batch_size, nstack, n) or (log n + 1, batch_size, nstack, n, 2)");
  auto d_input = grad.clone();
  auto d_twiddle = torch::zeros_like(twiddle);
  if (output.is_cuda()) {
    butterfly_multiply_untied_backward_cuda(twiddle, output, d_twiddle, d_input, increasing_stride);
    return {d_twiddle, d_input} ;
  }
  bool complex = grad.dim() == 4;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "butterfly_multiply_untied_backward", [&] {
    if (!complex) {
      const auto twiddle_a = twiddle.accessor<scalar_t, 5>();
      auto output_a = output.accessor<scalar_t, 4>();
      auto d_twiddle_a = d_twiddle.accessor<scalar_t, 5>();
      auto d_input_a = d_input.accessor<scalar_t, 3>();
      for (int64_t idx = log_n - 1; idx >= 0; --idx) {
        int64_t log_stride = increasing_stride ? idx : log_n - 1 - idx;
        int64_t stride = 1 << log_stride;
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t s = 0; s < nstack; ++s) {
            for (int64_t i = 0; i < n / 2; ++i) {
              int64_t low_order_bit = i % stride;
              int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
              const scalar_t twiddle_val[2][2] = {{twiddle_a[s][log_stride][i][0][0], twiddle_a[s][log_stride][i][0][1]},
                                                  {twiddle_a[s][log_stride][i][1][0], twiddle_a[s][log_stride][i][1][1]}};
              const scalar_t grad_val[2] = {d_input_a[b][s][pos], d_input_a[b][s][pos + stride]};
              d_input_a[b][s][pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
              d_input_a[b][s][pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
              const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
              d_twiddle_a[s][log_stride][i][0][0] += grad_val[0] * input_val[0];
              d_twiddle_a[s][log_stride][i][0][1] += grad_val[0] * input_val[1];
              d_twiddle_a[s][log_stride][i][1][0] += grad_val[1] * input_val[0];
              d_twiddle_a[s][log_stride][i][1][1] += grad_val[1] * input_val[1];
            }
          }
        }
      }
    } else {  // complex
      using complex_t = std::complex<scalar_t>;
      const auto twiddle_a = twiddle.accessor<scalar_t, 6>();
      const auto output_a = output.accessor<scalar_t, 5>();
      auto d_twiddle_a = d_twiddle.accessor<scalar_t, 6>();
      auto d_input_a = d_input.accessor<scalar_t, 4>();
      for (int64_t idx = log_n - 1; idx >= 0; --idx) {
        int64_t log_stride = increasing_stride ? idx : log_n - 1 - idx;
        int64_t stride = 1 << log_stride;
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t s = 0; s < nstack; ++s) {
            for (int64_t i = 0; i < n / 2; ++i) {
              int64_t low_order_bit = i % stride;
              int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
              const complex_t twiddle_val[2][2] =
                {{complex_t(twiddle_a[s][log_stride][i][0][0][0], twiddle_a[s][log_stride][i][0][0][1]),
                  complex_t(twiddle_a[s][log_stride][i][0][1][0], twiddle_a[s][log_stride][i][0][1][1])},
                 {complex_t(twiddle_a[s][log_stride][i][1][0][0], twiddle_a[s][log_stride][i][1][0][1]),
                  complex_t(twiddle_a[s][log_stride][i][1][1][0], twiddle_a[s][log_stride][i][1][1][1])}};
              const complex_t grad_val[2] =
                {complex_t(d_input_a[b][s][pos][0], d_input_a[b][s][pos][1]),
                 complex_t(d_input_a[b][s][pos + stride][0], d_input_a[b][s][pos + stride][1])};
              const complex_t d_input_val[2] =
                {std::conj(twiddle_val[0][0]) * grad_val[0] + std::conj(twiddle_val[1][0]) * grad_val[1],
                 std::conj(twiddle_val[0][1]) * grad_val[0] + std::conj(twiddle_val[1][1]) * grad_val[1]};
              d_input_a[b][s][pos][0] = d_input_val[0].real();
              d_input_a[b][s][pos][1] = d_input_val[0].imag();
              d_input_a[b][s][pos + stride][0] = d_input_val[1].real();
              d_input_a[b][s][pos + stride][1] = d_input_val[1].imag();
              const complex_t input_val[2] =
                {complex_t(output_a[idx][b][s][pos][0], output_a[idx][b][s][pos][1]),
                 complex_t(output_a[idx][b][s][pos + stride][0], output_a[idx][b][s][pos + stride][1])};
              const complex_t d_twiddle_val[2][2] =
                {{grad_val[0] * std::conj(input_val[0]), grad_val[0] * std::conj(input_val[1])},
                 {grad_val[1] * std::conj(input_val[0]), grad_val[1] * std::conj(input_val[1])}};
              d_twiddle_a[s][log_stride][i][0][0][0] += d_twiddle_val[0][0].real();
              d_twiddle_a[s][log_stride][i][0][0][1] += d_twiddle_val[0][0].imag();
              d_twiddle_a[s][log_stride][i][0][1][0] += d_twiddle_val[0][1].real();
              d_twiddle_a[s][log_stride][i][0][1][1] += d_twiddle_val[0][1].imag();
              d_twiddle_a[s][log_stride][i][1][0][0] += d_twiddle_val[1][0].real();
              d_twiddle_a[s][log_stride][i][1][0][1] += d_twiddle_val[1][0].imag();
              d_twiddle_a[s][log_stride][i][1][1][0] += d_twiddle_val[1][1].real();
              d_twiddle_a[s][log_stride][i][1][1][1] += d_twiddle_val[1][1].imag();
            }
          }
        }
      }
    }
  });
  return {d_twiddle, d_input} ;
}

at::Tensor butterfly_conv2d(const at::Tensor& twiddle, const at::Tensor& input,
  const size_t kernel_size, const size_t padding, bool increasing_stride, 
  bool return_intermediates) {
  /* Parameters:
        twiddle: (nstack, log n, n/2, 2, 2) where n = c_in  
        input: (b_in, c_in, h_in, w_in)
        kernel_size: int, size of convolution kernel, currently only supports square kernels 
        padding: amount of zero-padding around border of input  
        increasing_stride: whether to multiply with increasing stride (e.g. 1, 4, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 1).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed
    Returns:
        output: (b_in * h_out * w_out, nstack, c_in)
  */
  // Currently assuming convolution stride is 1
  const int b_in = input.size(0);
  const int c_in = input.size(1);
  // twiddle nstack = c_out/c_in * matrix batach 
  const int n = c_in; // rename to be consistent with dimension of butterfly
  const int c_out = twiddle.size(0) / (kernel_size*kernel_size) * c_in;
  const int h = input.size(2);
  const int w = input.size(3);
  const int log_n = int(log2((double) c_in)); 
  const auto bstack = twiddle.size(0);
  const int stack = kernel_size * kernel_size; 
  auto h_out = h + 2 * padding - (kernel_size - 1);
  auto w_out = w + 2 * padding - (kernel_size - 1);
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(input);
  AT_CHECK((twiddle.dim() == 5 && input.dim() == 4),
            "butterfly_conv2d: twiddle and input must have dimension 5,4");
  AT_CHECK(twiddle.device() == input.device(), "device of twiddle (", twiddle.device(), ") must match device of input (", input.device(), ")");
  AT_CHECK(twiddle.size(1) == log_n && twiddle.size(2) == n / 2 && twiddle.size(3) == 2 && twiddle.size(4) == 2, "butterfly_multiply_conv2d: twiddle must have shape (nstack, log n, n/2, 2, 2)");
  const int output_first_dim = return_intermediates ? log_n + 1 : 1;
  // return unfolded output 
  auto output = torch::zeros({output_first_dim, b_in*h_out*w_out, bstack, c_in},
    torch::dtype(input.dtype()).device(input.device()));
  if (!return_intermediates) {
    output = output.expand({log_n + 1, b_in*h_out*w_out, bstack, c_in});
  }
  butterfly_conv2d_cuda(twiddle, input, output, kernel_size, padding, h_out, w_out, increasing_stride, return_intermediates);
  return return_intermediates ? output : output[-1];
}

std::vector<at::Tensor> butterfly_conv2d_backward(const at::Tensor& grad, const at::Tensor& twiddle, 
  const at::Tensor& output, const size_t kernel_size, const size_t padding, 
  bool increasing_stride, const int b_in, const int c_in, 
  const int h_in, const int w_in) {
    /* Parameters:
         grad: (b_in * h_out * w_out, cin/cout * nstack, c_out) 
         twiddle: (nstack, log n, n / 2, 2, 2) where n = c_in
         output + intermediate values for backward: (log n + 1, b_in * h_out * w_out,
                                               cin/cout * nstack, c_out) 
         increasing_stride: whether the forward pass multiply was with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
          b_in: int, batch_size of input data 
          h_in: int, height of input data 
          w_in: int, width of input data
     Return:
         d_twiddle: (nstack, log n, n / 2, 2, 2) 
         d_input: (b_in, c_in, h_in, w_in) 
  */
  const int batch_size = grad.size(0);
  const int c_out = grad.size(2);
  const int bstack = grad.size(1) * c_out/c_in; 
  const int n = c_in; // rename to be consistent with dimension of butterfly
  const int log_n = int(log2((double) n));
  const int h_out = h_in + 2 * padding - (kernel_size - 1);
  const int w_out = w_in + 2 * padding - (kernel_size - 1);
  CHECK_DEVICE(grad);
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(output);
  AT_CHECK(grad.device() == twiddle.device() && twiddle.device() == output.device(), "device of grad (", grad.device(), ")twiddle (", twiddle.device(), "), and output (", output.device(), ") must match");
  AT_CHECK(twiddle.size(1) == log_n && twiddle.size(2) == n / 2 && twiddle.size(3) == 2 && twiddle.size(4) == 2, "butterfly_conv2d_backward: twiddle must have shape (nstack, log n, n/2, 2, 2) where n=c_in");
  AT_CHECK(output.size(0) == log_n + 1&& output.size(1) == batch_size && output.size(2) == bstack && output.size(3) == c_in, "butterfly_conv2d_backward: output must have shape (log n + 1, batch_size, nstack, n)");
  auto d_twiddle = torch::zeros_like(twiddle);
  auto d_input = torch::zeros({b_in, c_in, h_in, w_in},
    torch::dtype(grad.dtype()).device(grad.device()));
  butterfly_conv2d_backward_cuda(grad, twiddle, output, d_twiddle, d_input, 
                                 kernel_size, padding, h_out, w_out, 
                                 increasing_stride);
  return {d_twiddle, d_input};
}

at::Tensor butterfly_multiply_untied_svd(const at::Tensor& twiddle, const at::Tensor& input, bool increasing_stride, bool return_intermediates) {
  /* The twiddles uses SVD paramterization:
         [cos theta, -sin theta; sin theta, cos theta] [sigma_1, 0; 0, sigma_2] [cos phi, -sin phi; sin phi, cos phi]
     The order of storage is twiddle[2][2] = {{theta, phi}, {sigma_1, sigma_2}}.
     Only support real, not complex.
     Parameters:
         twiddle: (nstack, log n, n/2, 2, 2) if real or (nstack, log n, n/2, 2, 2, 2) if complex
         input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
         return_intermediates: whether to return just the output (i.e. computed in-place) or output
             and intermediate values for backward pass.
     Returns:
         if return_intermediates:
             output + intermediate values for backward pass: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
         else:
             output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
  */
  const auto batch_size = input.size(0);
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_CHECK((twiddle.dim() == 5 && input.dim() == 3),
           "butterfly_multiply_untied_svd: twiddle and input must have dimension 5,3");
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(input);
  AT_CHECK(twiddle.device() == input.device(), "device of twiddle (", twiddle.device(), ") must match device of input (", input.device(), ")");
  AT_CHECK(twiddle.size(0) == nstack && twiddle.size(1) == log_n && twiddle.size(2) == n / 2 && twiddle.size(3) == 2 && twiddle.size(4) == 2, "butterfly_multiply_untied_svd: twiddle must have shape (nstack, log n, n/2, 2, 2)");
  const int output_first_dim = return_intermediates ? log_n + 1 : 1;
  auto output = torch::empty({output_first_dim, batch_size, nstack, n}, torch::dtype(input.dtype()).device(input.device()));
  if (!return_intermediates) {
    output = output.expand({log_n + 1, batch_size, nstack, n});
  }
  output[0] = input;
  if (input.is_cuda()) {
    butterfly_multiply_untied_svd_cuda(twiddle, output, increasing_stride, return_intermediates);
    return return_intermediates ? output : output[-1];
  }
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_multiply_untied_svd", [&] {
    const auto twiddle_a = twiddle.accessor<scalar_t, 5>();
    auto output_a = output.accessor<scalar_t, 4>();
    for (int64_t idx = 0; idx <= log_n - 1; ++idx) {
      int64_t log_stride = increasing_stride ? idx : (log_n - 1 - idx);
      int64_t stride = 1 << log_stride;
      for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < nstack; ++s) {
          for (int64_t i = 0; i < n / 2; ++i) {
            int64_t low_order_bit = i % stride;
            int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
            const scalar_t twiddle_val[2][2] = {{twiddle_a[s][log_stride][i][0][0], twiddle_a[s][log_stride][i][0][1]},
                                                {twiddle_a[s][log_stride][i][1][0], twiddle_a[s][log_stride][i][1][1]}};
            const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
            const scalar_t sin_theta = std::sin(twiddle_val[0][0]), cos_theta = std::cos(twiddle_val[0][0]);
            const scalar_t sin_phi = std::sin(twiddle_val[0][1]), cos_phi = std::cos(twiddle_val[0][1]);
            scalar_t temp[2];
            std::tie(temp[0], temp[1]) = mult2x2(cos_phi, -sin_phi, sin_phi, cos_phi, input_val[0], input_val[1]);
            std::tie(temp[0], temp[1]) = mult2x2(cos_theta, -sin_theta, sin_theta, cos_theta,
                                                 temp[0] * twiddle_val[1][0], temp[1] * twiddle_val[1][1]);
            output_a[idx+1][b][s][pos] = temp[0];
            output_a[idx+1][b][s][pos + stride] = temp[1];
          }
        }
      }
    }
  });
  return return_intermediates ? output : output[-1];
}

std::vector<at::Tensor> butterfly_multiply_untied_svd_backward(const at::Tensor& grad, const at::Tensor& twiddle, const at::Tensor& output, bool increasing_stride) {
  /* The twiddles uses SVD paramterization:
     [cos theta, -sin theta; sin theta, cos theta] [sigma_1, 0; 0, sigma_2] [cos phi, -sin phi; sin phi, cos phi]
     The order of storage is twiddle[2][2] = {{theta, phi}, {sigma_1, sigma_2}}.
     Only support real, not complex.
     Parameters:
         grad: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
         twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
         output + untied_svd values for backward: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
         increasing_stride: whether the forward pass multiply was with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
     Return:
         d_twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
         d_input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
  */
  const auto batch_size = grad.size(0);
  const auto nstack = grad.size(1);
  const auto n = grad.size(2);
  const int log_n = int(log2((double) n));
  AT_CHECK((grad.dim() == 3 && twiddle.dim() == 5 && output.dim() == 4),
           "butterfly_multiply_untied_svd_backward: grad, twiddle, and output must have dimension 3,5,4");
  CHECK_DEVICE(grad);
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(output);
  AT_CHECK(grad.device() == twiddle.device() && twiddle.device() == output.device(), "device of grad (", grad.device(), ")twiddle (", twiddle.device(), "), and output (", output.device(), ") must match");
  AT_CHECK(twiddle.size(0) == nstack && twiddle.size(1) == log_n && twiddle.size(2) == n / 2 && twiddle.size(3) == 2 && twiddle.size(4) == 2, "butterfly_multiply_untied_svd_backward: twiddle must have shape (nstack, log n, n/2, 2, 2)");
  AT_CHECK(output.size(0) == log_n + 1 && output.size(1) == batch_size && output.size(2) == nstack && output.size(3) == n, "butterfly_multiply_untied_svd_backward: output must have shape (log n + 1, batch_size, nstack, n)");
  auto d_input = grad.clone();
  auto d_twiddle = torch::zeros_like(twiddle);
  if (output.is_cuda()) {
    butterfly_multiply_untied_svd_backward_cuda(twiddle, output, d_twiddle, d_input, increasing_stride);
    return {d_twiddle, d_input} ;
  }
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "butterfly_multiply_untied_svd_backward", [&] {
    const auto twiddle_a = twiddle.accessor<scalar_t, 5>();
    auto output_a = output.accessor<scalar_t, 4>();
    auto d_twiddle_a = d_twiddle.accessor<scalar_t, 5>();
    auto d_input_a = d_input.accessor<scalar_t, 3>();
    for (int64_t idx = log_n - 1; idx >= 0; --idx) {
      int64_t log_stride = increasing_stride ? idx : log_n - 1 - idx;
      int64_t stride = 1 << log_stride;
      for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < nstack; ++s) {
          for (int64_t i = 0; i < n / 2; ++i) {
            int64_t low_order_bit = i % stride;
            int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
            const scalar_t twiddle_val[2][2] = {{twiddle_a[s][log_stride][i][0][0], twiddle_a[s][log_stride][i][0][1]},
                                                {twiddle_a[s][log_stride][i][1][0], twiddle_a[s][log_stride][i][1][1]}};
            const scalar_t sin_theta = std::sin(twiddle_val[0][0]), cos_theta = std::cos(twiddle_val[0][0]);
            const scalar_t sin_phi = std::sin(twiddle_val[0][1]), cos_phi = std::cos(twiddle_val[0][1]);
            const scalar_t grad_val[2] = {d_input_a[b][s][pos], d_input_a[b][s][pos + stride]};
            scalar_t grad_temp_theta[2];
            std::tie(grad_temp_theta[0], grad_temp_theta[1])
              = mult2x2(cos_theta, sin_theta, -sin_theta, cos_theta, grad_val[0], grad_val[1]);
            const scalar_t grad_temp_diag[2] = {grad_temp_theta[0] * twiddle_val[1][0], grad_temp_theta[1] * twiddle_val[1][1]};
            std::tie(d_input_a[b][s][pos], d_input_a[b][s][pos + stride])
              = mult2x2(cos_phi, sin_phi, -sin_phi, cos_phi, grad_temp_diag[0], grad_temp_diag[1]);
            const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
            scalar_t input_temp_phi[2];
            std::tie(input_temp_phi[0], input_temp_phi[1])
              = mult2x2(cos_phi, -sin_phi, sin_phi, cos_phi, input_val[0], input_val[1]);
            const scalar_t input_temp_diag[2] = {input_temp_phi[0] * twiddle_val[1][0], input_temp_phi[1] * twiddle_val[1][1]};
            // d_theta
            d_twiddle_a[s][log_stride][i][0][0]
              += (grad_val[0] * input_temp_diag[0] + grad_val[1] * input_temp_diag[1]) * (-sin_theta)
              + (-grad_val[0] * input_temp_diag[1] + grad_val[1] * input_temp_diag[0]) * cos_theta;
            // d_sigma_1 and d_sigma_2
            d_twiddle_a[s][log_stride][i][1][0] += grad_temp_theta[0] * input_temp_phi[0];
            d_twiddle_a[s][log_stride][i][1][1] += grad_temp_theta[1] * input_temp_phi[1];
            // d_phi
            d_twiddle_a[s][log_stride][i][0][1]
              += (grad_temp_diag[0] * input_val[0] + grad_temp_diag[1] * input_val[1]) * (-sin_phi)
              + (-grad_temp_diag[0] * input_val[1] + grad_temp_diag[1] * input_val[0]) * cos_phi;
          }
        }
      }
    }
  });
  return {d_twiddle, d_input} ;
}

at::Tensor permutation_factor_even_odd_multiply(const at::Tensor& p, const at::Tensor& input) {
  /* Parameters:
         p: (1, )
         input: (batch_size, n) if real or (batch_size, n, 2) if complex
     Output:
         p input + (1 - p) input_permuted: (batch_size, n) if real or (batch_size, n, 2) if complex
  */
  auto output = torch::empty_like(input);
  if (input.is_cuda()) {
    AT_CHECK(p.is_cuda(), "permutation_factor_even_odd_multiply: Expected p to be CUDA tensor");
    permutation_factor_even_odd_multiply_cuda(p, input, output);
    return output;
  }
  AT_CHECK(!p.is_cuda(), "permutation_factor_even_odd_multiply: Expected p to be CPU tensor");
  const auto batch_size = input.size(0);
  const auto n = input.size(1);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_even_odd_multiply", [&] {
    const scalar_t p_a = p.accessor<scalar_t, 1>()[0];
    switch (input.dim()) {
      case 2: // real
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          output = output.view({batch_size, 2, n / 2});
          const auto input_a = input_folded.accessor<scalar_t, 3>();
          const auto permuted_input_a = permuted_input.accessor<scalar_t, 3>();
          auto output_a = output.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 2; ++i) {
              // Manually unrolling loop seems to be faster
              output_a[b][0][i] = (1 - p_a) * input_a[b][0][i] + p_a * permuted_input_a[b][0][i];
              output_a[b][1][i] = (1 - p_a) * input_a[b][1][i] + p_a * permuted_input_a[b][1][i];
            }
          }
          output = output.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          output = output.view({batch_size, 2, n / 2, 2});
          const auto input_a = input_folded.accessor<scalar_t, 4>();
          const auto permuted_input_a = permuted_input.accessor<scalar_t, 4>();
          auto output_a = output.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 2; ++i) {
              output_a[b][0][i][0] = (1 - p_a) * input_a[b][0][i][0] + p_a * permuted_input_a[b][0][i][0];
              output_a[b][0][i][1] = (1 - p_a) * input_a[b][0][i][1] + p_a * permuted_input_a[b][0][i][1];
              output_a[b][1][i][0] = (1 - p_a) * input_a[b][1][i][0] + p_a * permuted_input_a[b][1][i][0];
              output_a[b][1][i][1] = (1 - p_a) * input_a[b][1][i][1] + p_a * permuted_input_a[b][1][i][1];
            }
          }
          output = output.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_even_odd_multiply requires input dimension 2 or 3");
    }
  });
  return output;
}

std::vector<at::Tensor> permutation_factor_even_odd_multiply_backward(const at::Tensor& grad, const at::Tensor& p, const at::Tensor& input) {
  /* Parameters:
         grad: (batch_size, n) if real or (batch_size, n, 2) if complex
         p: (1, )
         input: (batch_size, n) if real or (batch_size, n, 2) if complex
     Output:
         d_p: (1, )
         d_input: (batch_size, n) if real or (batch_size, n, 2) if complex
  */
  const auto batch_size = grad.size(0);
  const auto n = grad.size(1);
  auto d_input = torch::empty_like(input);
  auto d_p = torch::zeros_like(p);
  if (input.is_cuda()) {
    AT_CHECK(grad.is_cuda() && p.is_cuda(), "permutation_factor_even_odd_multiply_backward: Expected grad and p to be CUDA tensor");
    // CUDA kernel will compute the expanded gradient of @p, then we'll call sum.
    // This is because I haven't figured out how to write efficient reduction kernel in CUDA.
    auto d_p_expanded = torch::empty({batch_size, n / 2}, torch::dtype(input.dtype()).device(input.device()));
    permutation_factor_even_odd_multiply_backward_cuda(grad, p, input, d_p_expanded, d_input);
    d_p[0] = d_p_expanded.sum();
    return {d_p, d_input};
  }
  AT_CHECK((!grad.is_cuda()) && (!p.is_cuda()), "permutation_factor_even_odd_multiply_backward: Expected grad and p to be CPU tensor");
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_even_odd_multiply_backward", [&] {
    const scalar_t p_a = p.accessor<scalar_t, 1>()[0];
    auto d_p_a = d_p.accessor<scalar_t, 1>();
    scalar_t d_p_temp = 0;
    switch (input.dim()) {
      case 2: // real
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          const auto grad_reshaped = grad.reshape({batch_size, 2, n / 2});
          const auto permuted_grad = grad.reshape({batch_size, 2, n / 2}).transpose(1, 2);
          const auto grad_folded = grad.reshape({batch_size, n / 2, 2});
          d_input = d_input.view({batch_size, n/ 2, 2});
          // Accessors
          const auto input_a = input_folded.accessor<scalar_t, 3>();
          const auto permuted_input_a = permuted_input.accessor<scalar_t, 3>();
          const auto grad_reshaped_a = grad_reshaped.accessor<scalar_t, 3>();
          const auto grad_a = grad_folded.accessor<scalar_t, 3>();
          const auto permuted_grad_a = permuted_grad.accessor<scalar_t, 3>();
          auto d_input_a = d_input.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 2; ++i) {
              d_p_temp += (permuted_input_a[b][0][i] - input_a[b][0][i]) * grad_reshaped_a[b][0][i]
                + (permuted_input_a[b][1][i] - input_a[b][1][i]) * grad_reshaped_a[b][1][i];
              d_input_a[b][i][0] = (1 - p_a) * grad_a[b][i][0] + p_a * permuted_grad_a[b][i][0];
              d_input_a[b][i][1] = (1 - p_a) * grad_a[b][i][1] + p_a * permuted_grad_a[b][i][1];
            }
          }
          d_input = d_input.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          const auto grad_reshaped = grad.reshape({batch_size, 2, n / 2, 2});
          const auto permuted_grad = grad.reshape({batch_size, 2, n / 2, 2}).transpose(1, 2);
          const auto grad_folded = grad.reshape({batch_size, n / 2, 2, 2});
          d_input = d_input.view({batch_size, n/ 2, 2, 2});
          // Accessors
          const auto input_a = input_folded.accessor<scalar_t, 4>();
          const auto permuted_input_a = permuted_input.accessor<scalar_t, 4>();
          const auto grad_reshaped_a = grad_reshaped.accessor<scalar_t, 4>();
          const auto grad_a = grad_folded.accessor<scalar_t, 4>();
          const auto permuted_grad_a = permuted_grad.accessor<scalar_t, 4>();
          auto d_input_a = d_input.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 2; ++i) {
              d_p_temp += (permuted_input_a[b][0][i][0] - input_a[b][0][i][0]) * grad_reshaped_a[b][0][i][0]
                + (permuted_input_a[b][0][i][1] - input_a[b][0][i][1]) * grad_reshaped_a[b][0][i][1]
                + (permuted_input_a[b][1][i][0] - input_a[b][1][i][0]) * grad_reshaped_a[b][1][i][0]
                + (permuted_input_a[b][1][i][1] - input_a[b][1][i][1]) * grad_reshaped_a[b][1][i][1];
              d_input_a[b][i][0][0] = (1 - p_a) * grad_a[b][i][0][0] + p_a * permuted_grad_a[b][i][0][0];
              d_input_a[b][i][0][1] = (1 - p_a) * grad_a[b][i][0][1] + p_a * permuted_grad_a[b][i][0][1];
              d_input_a[b][i][1][0] = (1 - p_a) * grad_a[b][i][1][0] + p_a * permuted_grad_a[b][i][1][0];
              d_input_a[b][i][1][1] = (1 - p_a) * grad_a[b][i][1][1] + p_a * permuted_grad_a[b][i][1][1];
            }
          }
          d_input = d_input.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_even_odd_multiply_backward requires input dimension 2 or 3");
    }
    d_p_a[0] = d_p_temp;
  });
  return {d_p, d_input};
}

at::Tensor permutation_factor_reverse_multiply(const at::Tensor& p, const at::Tensor& input) {
  /* Parameters:
         p: (2, )
         input: (batch_size, n) if real or (batch_size, n, 2) if complex
     Output:
         p input + (1 - p) input_reversed: (batch_size, n) if real or (batch_size, n, 2) if complex
  */
  auto output = torch::empty_like(input);
  AT_CHECK(input.size(1) > 2, "permutation_factor_reverse_multiply: n must be bigger than 2");
  if (input.is_cuda()) {
    AT_CHECK(p.is_cuda(), "permutation_factor_reverse_multiply: Expected p to be CUDA tensor");
    permutation_factor_reverse_multiply_cuda(p, input, output);
    return output;
  }
  AT_CHECK(!p.is_cuda(), "permutation_factor_reverse_multiply: Expected p to be CPU tensor");
  const auto batch_size = input.size(0);
  const auto n = input.size(1);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_reverse_multiply", [&] {
    const scalar_t p_a[2] = {p.accessor<scalar_t, 1>()[0], p.accessor<scalar_t, 1>()[1]};
    switch (input.dim()) {
      case 2: // real
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          output = output.view({batch_size, 2, n / 2});
          const auto input_a = input_folded.accessor<scalar_t, 3>();
          auto output_a = output.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 4; ++i) {
              // output_a[b][0][i] = (1 - p_a[0]) * input_a[b][0][i] + p_a[0] * input_a[b][0][n / 2 - 1 - i];
              const scalar_t in0[2] = {input_a[b][0][i], input_a[b][0][n / 2 - 1 - i]};
              output_a[b][0][i] = (1 - p_a[0]) * in0[0] + p_a[0] * in0[1];
              output_a[b][0][n / 2 - 1 - i] = p_a[0] * in0[0] + (1 - p_a[0]) * in0[1];
              // output_a[b][1][i] = (1 - p_a[1]) * input_a[b][1][i] + p_a[1] * input_a[b][1][n / 2 - 1 - i];
              const scalar_t in1[2] = {input_a[b][1][i], input_a[b][1][n / 2 - 1 - i]};
              output_a[b][1][i] = (1 - p_a[1]) * in1[0] + p_a[1] * in1[1];
              output_a[b][1][n / 2 - 1 - i] = p_a[1] * in1[0] + (1 - p_a[1]) * in1[1];
            }
          }
          output = output.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          output = output.view({batch_size, 2, n / 2, 2});
          const auto input_a = input_folded.accessor<scalar_t, 4>();
          auto output_a = output.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 4; ++i) {
              const scalar_t in00[2] = {input_a[b][0][i][0], input_a[b][0][n / 2 - 1 - i][0]};
              output_a[b][0][i][0] = (1 - p_a[0]) * in00[0] + p_a[0] * in00[1];
              output_a[b][0][n / 2 - 1 - i][0] = p_a[0] * in00[0] + (1 - p_a[0]) * in00[1];
              const scalar_t in01[2] = {input_a[b][0][i][1], input_a[b][0][n / 2 - 1 - i][1]};
              output_a[b][0][i][1] = (1 - p_a[0]) * in01[0] + p_a[0] * in01[1];
              output_a[b][0][n / 2 - 1 - i][1] = p_a[0] * in01[0] + (1 - p_a[0]) * in01[1];
              const scalar_t in10[2] = {input_a[b][1][i][0], input_a[b][1][n / 2 - 1 - i][0]};
              output_a[b][1][i][0] = (1 - p_a[1]) * in10[0] + p_a[1] * in10[1];
              output_a[b][1][n / 2 - 1 - i][0] = p_a[1] * in10[0] + (1 - p_a[1]) * in10[1];
              const scalar_t in11[2] = {input_a[b][1][i][1], input_a[b][1][n / 2 - 1 - i][1]};
              output_a[b][1][i][1] = (1 - p_a[1]) * in11[0] + p_a[1] * in11[1];
              output_a[b][1][n / 2 - 1 - i][1] = p_a[1] * in11[0] + (1 - p_a[1]) * in11[1];
            }
          }
          output = output.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_reverse_multiply requires input dimension 2 or 3");
    }
  });
  return output;
}

std::vector<at::Tensor> permutation_factor_reverse_multiply_backward(const at::Tensor& grad, const at::Tensor& p, const at::Tensor& input) {
  /* Parameters:
        grad: (batch_size, n) if real or (batch_size, n, 2) if complex
        p: (2, )
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
     Output:
        d_p: (2, )
        d_input: (batch_size, n) if real or (batch_size, n, 2) if complex
  */
  const auto batch_size = grad.size(0);
  const auto n = grad.size(1);
  AT_CHECK(n > 2, "permutation_factor_reverse_multiply_backward: n must be bigger than 2");
  auto d_input = torch::empty_like(input);
  if (input.is_cuda()) {
    AT_CHECK(grad.is_cuda() && p.is_cuda(), "permutation_factor_reverse_multiply_backward: Expected grad and p to be CUDA tensor");
    // CUDA kernel will compute the expanded gradient of @p, then we'll call sum.
    // This is because I haven't figured out how to write efficient reduction kernel in CUDA.
    auto d_p_expanded = torch::empty({2, batch_size, n / 4}, torch::dtype(input.dtype()).device(input.device()));
    permutation_factor_reverse_multiply_backward_cuda(grad, p, input, d_p_expanded, d_input);
    return {d_p_expanded.sum(/*dim=*/{1, 2}), d_input};
  }
  AT_CHECK((!grad.is_cuda()) && (!p.is_cuda()), "permutation_factor_reverse_multiply_backward: Expected grad and p to be CPU tensor");
  auto d_p = torch::zeros_like(p);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_reverse_multiply_backward", [&] {
    const scalar_t p_a[2] = {p.accessor<scalar_t, 1>()[0], p.accessor<scalar_t, 1>()[1]};
    auto d_p_a = d_p.accessor<scalar_t, 1>();
    scalar_t d_p_temp[2] = {0, 0};
    switch (input.dim()) {
      case 2: // real
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          const auto grad_folded = grad.reshape({batch_size, 2, n / 2});
          d_input = d_input.view({batch_size, 2, n / 2});
          // Accessors
          const auto input_a = input_folded.accessor<scalar_t, 3>();
          const auto grad_a = grad_folded.accessor<scalar_t, 3>();
          auto d_input_a = d_input.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 4; ++i) {
              const scalar_t in0[2] = {input_a[b][0][i], input_a[b][0][n / 2 - 1 - i]};
              const scalar_t g0[2] = {grad_a[b][0][i], grad_a[b][0][n / 2 - 1 - i]};
              d_p_temp[0] += (in0[1] - in0[0]) * (g0[0] - g0[1]);
              d_input_a[b][0][i] = (1 - p_a[0]) * g0[0] + p_a[0] * g0[1];
              d_input_a[b][0][n / 2 - 1 - i] = p_a[0] * g0[0] + (1 - p_a[0]) * g0[1];
              const scalar_t in1[2] = {input_a[b][1][i], input_a[b][1][n / 2 - 1 - i]};
              const scalar_t g1[2] = {grad_a[b][1][i], grad_a[b][1][n / 2 - 1 - i]};
              d_p_temp[1] += (in1[1] - in1[0]) * (g1[0] - g1[1]);
              d_input_a[b][1][i] = (1 - p_a[1]) * g1[0] + p_a[1] * g1[1];
              d_input_a[b][1][n / 2 - 1 - i] = p_a[1] * g1[0] + (1 - p_a[1]) * g1[1];
            }
          }
          d_input = d_input.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          const auto grad_folded = grad.reshape({batch_size, 2, n / 2, 2});
          d_input = d_input.view({batch_size, 2, n / 2, 2});
          // Accessors
          const auto input_a = input_folded.accessor<scalar_t, 4>();
          const auto grad_a = grad_folded.accessor<scalar_t, 4>();
          auto d_input_a = d_input.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 4; ++i) {
              const scalar_t in00[2] = {input_a[b][0][i][0], input_a[b][0][n / 2 - 1 - i][0]};
              const scalar_t g00[2] = {grad_a[b][0][i][0], grad_a[b][0][n / 2 - 1 - i][0]};
              d_p_temp[0] += (in00[1] - in00[0]) * (g00[0] - g00[1]);
              d_input_a[b][0][i][0] = (1 - p_a[0]) * g00[0] + p_a[0] * g00[1];
              d_input_a[b][0][n / 2 - 1 - i][0] = p_a[0] * g00[0] + (1 - p_a[0]) * g00[1];
              const scalar_t in01[2] = {input_a[b][0][i][1], input_a[b][0][n / 2 - 1 - i][1]};
              const scalar_t g01[2] = {grad_a[b][0][i][1], grad_a[b][0][n / 2 - 1 - i][1]};
              d_p_temp[0] += (in01[1] - in01[0]) * (g01[0] - g01[1]);
              d_input_a[b][0][i][1] = (1 - p_a[0]) * g01[0] + p_a[0] * g01[1];
              d_input_a[b][0][n / 2 - 1 - i][1] = p_a[0] * g01[0] + (1 - p_a[0]) * g01[1];
              const scalar_t in10[2] = {input_a[b][1][i][0], input_a[b][1][n / 2 - 1 - i][0]};
              const scalar_t g10[2] = {grad_a[b][1][i][0], grad_a[b][1][n / 2 - 1 - i][0]};
              d_p_temp[1] += (in10[1] - in10[0]) * (g10[0] - g10[1]);
              d_input_a[b][1][i][0] = (1 - p_a[1]) * g10[0] + p_a[1] * g10[1];
              d_input_a[b][1][n / 2 - 1 - i][0] = p_a[1] * g10[0] + (1 - p_a[1]) * g10[1];
              const scalar_t in11[2] = {input_a[b][1][i][1], input_a[b][1][n / 2 - 1 - i][1]};
              const scalar_t g11[2] = {grad_a[b][1][i][1], grad_a[b][1][n / 2 - 1 - i][1]};
              d_p_temp[1] += (in11[1] - in11[0]) * (g11[0] - g11[1]);
              d_input_a[b][1][i][1] = (1 - p_a[1]) * g11[0] + p_a[1] * g11[1];
              d_input_a[b][1][n / 2 - 1 - i][1] = p_a[1] * g11[0] + (1 - p_a[1]) * g11[1];
            }
          }
          d_input = d_input.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_reverse_multiply_backward requires input dimension 2 or 3");
    }
    d_p_a[0] = d_p_temp[0];
    d_p_a[1] = d_p_temp[1];
  });
  return {d_p, d_input};
}

// at::Tensor flip(const at::Tensor& input) {
//   auto output = torch::empty_like(input);
//   auto input_neg_strides = torch::from_blob(input.data<float>() + input.size(0) - 1, input.sizes(), {-1});
//   output.copy_(input_neg_strides);
//   return output;
// }

void real_to_complex_strides(at::Tensor& x) {
  /*
    Change from real strides (array of size (..., 2)) to complex strides (array of size (...)).
   */
  AT_CHECK(x.size(-1) == 2 && x.stride(-1) == 1, "x must have last dimension == 2, which must be contiguous");
  auto strides_ptr = const_cast<int64_t*>(x.strides().data());
  for (int64_t i = 0; i < x.dim() - 1; ++i) {
    strides_ptr[i] /= 2;
  }
}

void complex_to_real_strides(at::Tensor& x) {
  /*
    Change from complex strides (array of size (...)) to real strides (array of size (..., 2)).
  */
  AT_CHECK(x.size(-1) == 2 && x.stride(-1) == 1, "x must have last dimension == 2, which must be contiguous");
  auto strides_ptr = const_cast<int64_t*>(x.strides().data());
  for (int64_t i = 0; i < x.dim() - 1; ++i) {
    strides_ptr[i] *= 2;
  }
}

// #define COMPLEX_ACCESSOR(x, dim, name)           \
//   real_to_complex_strides(x); \
//   return AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), name [&] { \
//     auto ptr = reinterpret_cast<std::complex<scalar_t>*>(x.data<scalar_t>()); \
//     return at::TensorAccessor<std::complex<scalar_t>, dim>(ptr, x.sizes().data(), x.strides().data()); \
//   })

void complex_test(at::Tensor& input) {
  auto ptr = reinterpret_cast<std::complex<float>*>(input.data<float>());
  ptr[0] = std::complex<float>(0.0, 0.0);
  // auto output = at::CPU(at::kComplexFloat).tensorfromBlob(ptr, {2}); // doesn't compile
  // int64_t sizes[1] = {input.size(0)};
  // int64_t strides[1] = {input.stride(0)};
  // auto input_a = at::TensorAccessor<std::complex<float>, 1>(ptr, input.sizes().data(), input.strides().data());
  real_to_complex_strides(input);
  // auto input_a = AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), name [&] {
  //   auto ptr = reinterpret_cast<std::complex<scalar_t>*>(input.data<scalar_t>()); \
  //   return at::TensorAccessor<std::complex<scalar_t>, 1>(ptr, input.sizes().data(), input.strides().data());
  // });
  complex_to_real_strides(input);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("butterfly_factor_multiply", &butterfly_factor_multiply, "Butterfly factor multiply forward");
  m.def("butterfly_factor_multiply_backward", &butterfly_factor_multiply_backward, "Butterfly factor multiply backward");
  m.def("butterfly_multiply_inplace", &butterfly_multiply_inplace, "Butterfly factor multiply inplace forward");
  m.def("butterfly_multiply_inplace_backward", &butterfly_multiply_inplace_backward, "Butterfly factor multiply inplace backward");
  m.def("butterfly_multiply_intermediate", &butterfly_multiply_intermediate, "Butterfly factor multiply intermediate forward");
  m.def("butterfly_multiply_intermediate_backward", &butterfly_multiply_intermediate_backward, "Butterfly factor multiply intermediate backward");
  m.def("butterfly_multiply_untied", &butterfly_multiply_untied, "Butterfly factor multiply untied forward");
  m.def("butterfly_multiply_untied_backward", &butterfly_multiply_untied_backward, "Butterfly factor multiply untied backward");
  m.def("butterfly_conv2d", &butterfly_conv2d, "Butterfly conv2d forward");
  m.def("butterfly_conv2d_backward", &butterfly_conv2d_backward, "Butterfly conv2d backward");
  m.def("butterfly_multiply_untied_svd", &butterfly_multiply_untied_svd, "Butterfly factor multiply untied SVD forward");
  m.def("butterfly_multiply_untied_svd_backward", &butterfly_multiply_untied_svd_backward, "Butterfly factor multiply untied SVD backward");
  m.def("permutation_factor_even_odd_multiply", &permutation_factor_even_odd_multiply, "Permutation factor (even odd) multiply forward");
  m.def("permutation_factor_even_odd_multiply_backward", &permutation_factor_even_odd_multiply_backward, "Permutation factor (even odd) multiply backward");
  m.def("permutation_factor_reverse_multiply", &permutation_factor_reverse_multiply, "Permutation factor (reverse) multiply forward");
  m.def("permutation_factor_reverse_multiply_backward", &permutation_factor_reverse_multiply_backward, "Permutation factor (even odd) multiply backward");
  m.def("complex_test", &complex_test, "complex_test");
}
