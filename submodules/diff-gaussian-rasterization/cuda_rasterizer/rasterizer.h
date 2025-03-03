/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int ref_D, int M, int ref_M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* ref_shs,//new
			const float* ref_colors_precomp,//new
			const float* opacities,
			const float* ref_opacities,//new
			const float* betas,//new
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* trans_weights,//new
			float* comp_ref_color,//new
			float* comp_trans_color,//new
			float* out_color,
			float* depth,
			bool antialiasing,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int ref_D, int M, int ref_M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* ref_shs,//new
			const float* ref_colors_precomp,//new
			const float* betas, //new
			const float* opacities,
			const float* ref_opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			// const float* dL_dpix,
			const float* trans_weights, 
			const float* comp_ref_color,
			const float* dL_dpix_ref_map,
			const float* dL_dpix_ref_color,
			const float* dL_dpix_trans_color,
			//new
			const float* dL_invdepths,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_drefopacity,//new
			float* dL_dbeta,//new
			float* dL_dtranscolor,//new
			float* dL_drefcolor,//new
			// float* dL_dcolor,
			float* dL_dinvdepth,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_drefsh, //new
			float* dL_dscale,
			float* dL_drot,
			bool antialiasing,
			bool debug);
	};
};

#endif
