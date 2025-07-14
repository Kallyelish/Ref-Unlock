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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

__device__ void refcomputeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
				
				if (deg > 3)
				{
					float dRGBdsh16 = SH_C4[0] * xy * (xx - yy);
					float dRGBdsh17 = SH_C4[1] * yz * (3.f * xx - yy);
					float dRGBdsh18 = SH_C4[2] * xy * (7.f * zz - 1.f);
					float dRGBdsh19 = SH_C4[3] * yz * (7.f * zz - 3.f);
					float dRGBdsh20 = SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f);
					float dRGBdsh21 = SH_C4[5] * xz * (7.f * zz - 3.f);
					float dRGBdsh22 = SH_C4[6] * (xx - yy) * (7.f * zz - 1.f);
					float dRGBdsh23 = SH_C4[7] * xz * (xx - 3.f * yy);
					float dRGBdsh24 = SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy));

					dL_dsh[16] = dRGBdsh16 * dL_dRGB;
					dL_dsh[17] = dRGBdsh17 * dL_dRGB;
					dL_dsh[18] = dRGBdsh18 * dL_dRGB;
					dL_dsh[19] = dRGBdsh19 * dL_dRGB;
					dL_dsh[20] = dRGBdsh20 * dL_dRGB;
					dL_dsh[21] = dRGBdsh21 * dL_dRGB;
					dL_dsh[22] = dRGBdsh22 * dL_dRGB;
					dL_dsh[23] = dRGBdsh23 * dL_dRGB;
					dL_dsh[24] = dRGBdsh24 * dL_dRGB;

					dRGBdx += (
						SH_C4[0] * sh[16] * (3.f * y * xx - y * yy) +
						SH_C4[1] * sh[17] * yz * 3.f * 2.f * x +
						SH_C4[2] * sh[18] * y * (7.f * zz - 1.f) +
						SH_C4[5] * sh[21] * z * (7.f * zz - 3.f) +
						SH_C4[6] * sh[22] * 2.f * x * (7.f * zz - 1.f) +
						SH_C4[7] * sh[23] * 3.f * z * (xx - yy) +
						SH_C4[8] * sh[24] * (4.f * xx * x - 12.f * x * yy)
					);

					dRGBdy += (
						SH_C4[0] * sh[16] * x * (xx - 3.f * yy) +
						SH_C4[1] * sh[17] * 3.f * z * (xx - yy) +
						SH_C4[2] * sh[18] * x * (7.f * zz - 1.f) +
						SH_C4[3] * sh[19] * z * (7.f * zz - 3.f) +
						SH_C4[6] * sh[22] * (-2.f) * y * (7.f * zz - 1.f) +
						SH_C4[7] * sh[23] * xz * (- 3.f) * 2.f * y +
						SH_C4[8] * sh[24] * 4.f * y * (yy - 3.f * xx)
					);

					dRGBdz += (
						SH_C4[1] * sh[17] * y * (3.f * xx - yy) + 
						SH_C4[2] * sh[18] * xy * 7.f * 2.f * z +
						SH_C4[3] * sh[19] * 3.f * y * (7.f * zz - 1.f) +
						SH_C4[4] * sh[20] * 20.f * z * (7.f * zz - 3.f) + 
						SH_C4[5] * sh[21] * 3.f * x * (7.f * zz - 1.f) +
						SH_C4[6] * sh[22] * (xx - yy) * 7.f * 2.f * z +
						SH_C4[7] * sh[23] * x * (xx - 3.f * yy)
					);

					if(deg > 4)
					{
						float dRGBdsh25 = SH_C5[0] * y * (5.f * xx * xx - 10.f * xx * yy + yy * yy);
						float dRGBdsh26 = SH_C5[1] * 4.f * xy * (xx - yy) * z;
						float dRGBdsh27 = SH_C5[2] * y * (3.f * xx - yy) * (9.f * zz - 1.f);
						float dRGBdsh28 = SH_C5[3] * 2.f * xy * z * (3.f * zz - 1.f);
						float dRGBdsh29 = SH_C5[4] * y * (21.f * zz * zz - 14.f * zz + 1.f);
						float dRGBdsh30 = SH_C5[5] * z * (63.f * zz * zz - 70.f * zz + 15.f);
						float dRGBdsh31 = SH_C5[6] * x * (21.f * zz * zz - 14.f * zz + 1.f);
						float dRGBdsh32 = SH_C5[7] * (xx - yy) * z * (3.f * zz - 1.f);
						float dRGBdsh33 = SH_C5[8] * x * (xx - 3.f * yy) * (9.f * zz - 1.f);
						float dRGBdsh34 = SH_C5[9] * (xx * xx - 6.f * xx * yy + yy * yy) * z;
						float dRGBdsh35 = SH_C5[10] * x * (xx * xx - 10.f * xx * yy + 5.f * yy * yy);

						dL_dsh[25] = dRGBdsh25 * dL_dRGB;
						dL_dsh[26] = dRGBdsh26 * dL_dRGB;
						dL_dsh[27] = dRGBdsh27 * dL_dRGB;
						dL_dsh[28] = dRGBdsh28 * dL_dRGB;
						dL_dsh[29] = dRGBdsh29 * dL_dRGB;
						dL_dsh[30] = dRGBdsh30 * dL_dRGB;
						dL_dsh[31] = dRGBdsh31 * dL_dRGB;
						dL_dsh[32] = dRGBdsh32 * dL_dRGB;
						dL_dsh[33] = dRGBdsh33 * dL_dRGB;
						dL_dsh[34] = dRGBdsh34 * dL_dRGB;
						dL_dsh[35] = dRGBdsh35 * dL_dRGB;

						dRGBdx += (
							SH_C5[0] * sh[25] * y * (5.f * 4.f * x * xx - 10.f * 2.f * x * yy) +
							SH_C5[1] * sh[26] * 4.f * z * (3.f * xx * y - y * yy) +
							SH_C5[2] * sh[27] * 6.f * xy * (9.f * zz - 1.f) +
							SH_C5[3] * sh[28] * 2.f * yz * (3.f * zz - 1.f) +
							SH_C5[6] * sh[31] * (21.f * zz * zz - 14.f * zz + 1.f) +
							SH_C5[7] * sh[32] * 2.f * xz * (3.f * zz - 1.f) +
							SH_C5[8] * sh[33] * 3.f * (xx - yy)* (9.f * zz - 1.f) +
							SH_C5[9] * sh[34] * (4.f * x * xx - 12.f * x * yy) * z +
							SH_C5[10] * sh[35] * (5.f * xx * xx - 30.f * xx * yy + 5.f * yy * yy)
						);

						dRGBdy += (
							SH_C5[0] * sh[25] * (5.f * xx * xx - 30.f * xx * yy + 5.f * yy * yy) +
							SH_C5[1] * sh[26] * 4.f * xz * (xx - 3.f * yy) +
							SH_C5[2] * sh[27] * 3.f * (xx - yy) * (9.f * zz - 1.f) +
							SH_C5[3] * sh[28] * 2.f * xz * (3.f * zz - 1.f) +
							SH_C5[4] * sh[29] * (21.f * zz * zz - 14.f * zz + 1.f) +
							SH_C5[5] * sh[30] * z * (63.f * zz * zz - 70.f * zz + 15.f) +
							SH_C5[7] * sh[32] * (-2.f) * yz * (3.f * zz - 1.f) +
							SH_C5[8] * sh[33] * (-6.f) * xy * (9.f * zz - 1.f) +
							SH_C5[9] * sh[34] * 4.f * yz * (yy - 3.f * xx) +
							SH_C5[10] * sh[35] * 20.f * xy * (yy - xx)
						);

						dRGBdz += (
							SH_C5[1] * sh[26] * 4.f * xy * (xx - yy) +
							SH_C5[2] * sh[27] * (3.f * xx - yy) * 18.f * yz +
							SH_C5[3] * sh[28] * 2.f * xy * (9.f * zz - 1.f) +
							SH_C5[4] * sh[29] * 28.f * yz * (3.f * zz - 1.f) +
							SH_C5[5] * sh[30] * 15.f * (21.f * zz * zz - 14.f * zz + 1.f) +
							SH_C5[6] * sh[31] * 28.f * xz * (3.f * zz - 1.f) +
							SH_C5[7] * sh[32] * (xx - yy) * (6.f * zz - 1.f) +
							SH_C5[8] * sh[33] * (xx - 3.f * yy) * 18.f * xz + 
							SH_C5[9] * sh[34] * (xx * xx - 6.f * xx * yy + yy * yy)
						);
					}
				}		
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* opacities,
	const float* ref_opacities,
	const float* trans_weights,
	const float* dL_dconics,
	float* dL_dopacity,
	float* dL_drefopacity,
	const float* dL_dinvdepth,
	float3* dL_dmeans,
	float* dL_dcov,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float c_xx = cov2D[0][0];
	float c_xy = cov2D[0][1];
	float c_yy = cov2D[1][1];
	
	constexpr float h_var = 0.3f;
	float d_inside_root = 0.f;
	if(antialiasing)
	{
		const float det_cov = c_xx * c_yy - c_xy * c_xy;
		c_xx += h_var;
		c_yy += h_var;
		const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
		const float h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability
		const float dL_dopacity_v = dL_dopacity[idx];
		const float dL_drefopacity_v = dL_drefopacity[idx];
		const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx] * (1 - trans_weights[idx]) + dL_drefopacity_v *ref_opacities[idx] * trans_weights[idx];
		dL_dopacity[idx] = dL_dopacity_v * h_convolution_scaling;
		dL_drefopacity[idx] = dL_drefopacity_v * h_convolution_scaling;
		d_inside_root = (det_cov / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
	} 
	else
	{
		c_xx += h_var;
		c_yy += h_var;
	}
	
	float dL_dc_xx = 0;
	float dL_dc_xy = 0;
	float dL_dc_yy = 0;
	if(antialiasing)
	{
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdx
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdz
		const float x = c_xx;
		const float y = c_yy;
		const float z = c_xy;
		const float w = h_var;
		const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
		const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
		const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
		const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
		dL_dc_xx = dL_dx;
		dL_dc_yy = dL_dy;
		dL_dc_xy = dL_dz;
	}
	
	float denom = c_xx * c_yy - c_xy * c_xy;

	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		
		dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_dc_xx + T[0][0] * T[1][0] * dL_dc_xy + T[1][0] * T[1][0] * dL_dc_yy);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_dc_xx + T[0][1] * T[1][1] * dL_dc_xy + T[1][1] * T[1][1] * dL_dc_yy);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_dc_xx + T[0][2] * T[1][2] * dL_dc_xy + T[1][2] * T[1][2] * dL_dc_yy);
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_dc_xx + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][1] * dL_dc_yy;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_dc_xx + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][2] * dL_dc_yy;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_dc_xx + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_dc_xy + 2 * T[1][1] * T[1][2] * dL_dc_yy;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xx +
	(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xx +
	(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xx +
	(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_xy;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_yy +
	(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_yy +
	(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_yy +
	(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xy;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;
	// Account for inverse depth gradients
	if (dL_dinvdepth)
	dL_dtz -= dL_dinvdepth[idx] / (t.z * t.z);


	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int ref_D, int M, int ref_M,
	const float3* means,
	const int* radii,
	const float* shs,
	const float* ref_shs, //new
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	// const float* view, //new
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	// float* dL_dcolor,
	float* dL_dtranscolor,//new
	float* dL_drefcolor,//new
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_drefsh, //new
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dopacity)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		refcomputeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dtranscolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	if (ref_shs)
		refcomputeColorFromSH(idx, ref_D, ref_M, (glm::vec3*)means, *campos, ref_shs, clamped, (glm::vec3*)dL_drefcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_drefsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float4* __restrict__ conic_ref_opacity,//new
	// const float* __restrict__ colors,
	//new_b
	const float* __restrict__ trans_colors,
	const float* __restrict__ ref_colors,
	const float* __restrict__ betas,
	//new_e
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	//new_b
	const float* __restrict__ final_ref_Ts,
	const float* __restrict__ trans_weights,
	const float* __restrict__ comp_ref_color,
	//new_e
	const uint32_t* __restrict__ n_contrib,
	// const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixels_ref_map,
	const float* __restrict__ dL_dpixels_ref_color,
	const float* __restrict__ dL_dpixels_trans_color,
	//new_e
	const float* __restrict__ dL_invdepths,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_drefopacity,//new
	float* __restrict__ dL_dbeta, //new
	// float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dtranscolors, 
	float* __restrict__ dL_drefcolors,
	float* __restrict__ dL_dinvdepths
)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float4 collected_conic_ref_opacity[BLOCK_SIZE];
	__shared__ float collected_trans_colors[C * BLOCK_SIZE];
	__shared__ float collected_ref_colors[C * BLOCK_SIZE];
	__shared__ float collected_betas[BLOCK_SIZE];
	// __shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];


	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	const float ref_T_final = inside ? final_ref_Ts[pix_id] : 0;
	float T = T_final;
	float ref_T = ref_T_final;
	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	// float accum_rec[C] = { 0 };
	// float dL_dpixel[C];
	float accum_trans_rec[C] = { 0 };
	float accum_ref_rec[C] = { 0 };
	float accum_beta_rec = 0;
	float dL_dpixel_ref_color[C];
	float dL_dpixel_trans_color[C];
	float dL_dpixel_ref_map = 0;
	
	float dL_invdepth;
	float accum_invdepth_rec = 0;
	if (inside)
	{
		for (int i = 0; i < C; i++){
			dL_dpixel_ref_color[i] = dL_dpixels_ref_color[i * H * W + pix_id];
			dL_dpixel_trans_color[i] = dL_dpixels_trans_color[i * H * W + pix_id];
		}
		dL_dpixel_ref_map = dL_dpixels_ref_map[pix_id];
			// dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		if(dL_invdepths)
		dL_invdepth = dL_invdepths[pix_id];
	}

	// float last_alpha = 0;
	// float last_color[C] = { 0 };
	float last_trans_alpha = 0;
	float last_ref_alpha = 0;
	float last_beta = 0;
	float last_trans_color[C] = { 0 };
	float last_ref_color[C] = { 0 };
	
	float last_invdepth = 0;


	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_conic_ref_opacity[block.thread_rank()] = conic_ref_opacity[coll_id];
			collected_betas[block.thread_rank()] = betas[coll_id];
			for (int i = 0; i < C; i++){
				collected_trans_colors[i * BLOCK_SIZE + block.thread_rank()] = trans_colors[coll_id * C + i];
				collected_ref_colors[i * BLOCK_SIZE + block.thread_rank()] = ref_colors[coll_id * C + i];
			}
				// collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];

			if(dL_invdepths)
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float4 con_ref_o = collected_conic_ref_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			const float power_ref = -0.5f * (con_ref_o.x * d.x * d.x + con_ref_o.z * d.y * d.y) - con_ref_o.y * d.x * d.y;
			if (power > 0.0f || power_ref > 0.0f)
			// if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float G_ref = exp(power_ref);
			const float alpha = min(0.99f, con_o.w * G);
			const float ref_alpha = min(0.99f, con_ref_o.w * G_ref);
			if (alpha < 1.0f / 255.0f || ref_alpha < 1.0f / 255.0f)
			// if (alpha < 1.0f / 255.0f)
				continue;
			
			const float beta = collected_betas[j];
			
			T = T / (1.f - alpha);
			ref_T = ref_T / (1.f - ref_alpha);
			// const float dchannel_dcolor = alpha * T;
			const float dchannel_dtranscolor = alpha * T;
			const float dchannel_drefcolor =  ref_alpha * ref_T;
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			// float dL_dalpha = 0.0f;
			float dL_dtransalpha = 0.0f;
			float dL_drefalpha = 0.0f;
			float dL_dchannelbeta = 0.0f;

			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float trans_c = collected_trans_colors[ch * BLOCK_SIZE + j];
				const float ref_c = collected_ref_colors[ch * BLOCK_SIZE + j];
				
				// const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				// accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				// last_color[ch] = c;
				accum_trans_rec[ch] = last_trans_alpha * (last_trans_color[ch]) + (1.f - last_trans_alpha) * accum_trans_rec[ch];
				accum_ref_rec[ch] = last_ref_alpha * last_ref_color[ch] + (1.f - last_ref_alpha) * accum_ref_rec[ch];
				last_trans_color[ch] = trans_c;
				last_ref_color[ch] = ref_c;

				// const float dL_dchannel = dL_dpixel[ch];
				// dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				const float dL_dchannel_trans = dL_dpixel_trans_color[ch];
				const float dL_dchannel_ref = dL_dpixel_ref_color[ch];
				dL_dtransalpha += (trans_c - accum_trans_rec[ch]) * dL_dchannel_trans;
				dL_drefalpha += (ref_c - accum_ref_rec[ch]) * dL_dchannel_ref;
				
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				// atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				atomicAdd(&(dL_dtranscolors[global_id * C + ch]), dchannel_dtranscolor * dL_dchannel_trans);
				atomicAdd(&(dL_drefcolors[global_id * C + ch]), dchannel_drefcolor * dL_dchannel_ref);
			
			}
			// Propagate gradients from inverse depth to alphaas and
			// per Gaussian inverse depths
			if (dL_dinvdepths)
			{
			const float invd = 1.f / collected_depths[j];
			accum_invdepth_rec = last_trans_alpha * last_invdepth + (1.f - last_trans_alpha) * accum_invdepth_rec;
			last_invdepth = invd;
			dL_dtransalpha += (invd - accum_invdepth_rec) * dL_invdepth;
			atomicAdd(&(dL_dinvdepths[global_id]), dchannel_dtranscolor * dL_invdepth);
			}

			// dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			// last_alpha = alpha;
			accum_beta_rec = last_trans_alpha * last_beta + (1.f - last_trans_alpha) * accum_beta_rec;
			dL_dtransalpha += (beta - accum_beta_rec) * dL_dpixel_ref_map;
			dL_dtransalpha *= T;
			dL_drefalpha *= ref_T;
			dL_dchannelbeta += dchannel_dtranscolor * dL_dpixel_ref_map; 
			// Update last alpha (to be used in the next iteration)
			last_trans_alpha = alpha;
			last_ref_alpha = ref_alpha;
			last_beta = beta;
			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel_trans_color[i];
				// bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			// dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
			
			dL_dtransalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dtransalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			const float dL_dG_ref = con_ref_o.w * dL_drefalpha;
			const float g_refdx = G_ref * d.x;
			const float g_refdy = G_ref * d.y;
			const float dG_ref_ddelx = -g_refdx * con_ref_o.x - g_refdy * con_ref_o.y;
			const float dG_ref_ddely = -g_refdy * con_ref_o.z - g_refdx * con_ref_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG_ref * dG_ref_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG_ref * dG_ref_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * g_refdx * d.x * dL_dG_ref);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * g_refdx * d.y * dL_dG_ref);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * g_refdy * d.y * dL_dG_ref);

			// Update gradients w.r.t. opacity of the Gaussian
			// atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dtransalpha);
			atomicAdd(&(dL_drefopacity[global_id]), G_ref * dL_drefalpha);
			atomicAdd(&(dL_dbeta[global_id]), dL_dchannelbeta);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int ref_D, int M, int ref_M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const float* ref_shs,//new
	const bool* clamped,
	const float* opacities,
	const float* ref_opacities,
	const float* trans_weights,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_dinvdepth,
	float* dL_dopacity,
	float* dL_drefopacity,
	glm::vec3* dL_dmean3D,
	// float* dL_dcolor,
	float* dL_dtranscolor,//new
	float* dL_drefcolor,//new
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_drefsh,//new
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	bool antialiasing)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		opacities,
		ref_opacities,
		trans_weights,
		dL_dconic,
		dL_dopacity,
		dL_drefopacity,
		dL_dinvdepth,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		antialiasing);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, ref_D, M, ref_M,
		(float3*)means3D,
		radii,
		shs,
		ref_shs, 
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		// dL_dcolor,
		dL_dtranscolor,
		dL_drefcolor,
		dL_dcov3D,
		dL_dsh,
		dL_drefsh,
		dL_dscale,
		dL_drot,
		dL_dopacity);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float4* conic_ref_opacity,//new
	const float* colors,
	const float* ref_colors,//new
	const float* betas,//new
	const float* depths,
	const float* final_Ts,
	const float* final_ref_Ts,//new
	const float* trans_weights,//new
	const float* comp_ref_color,//new
	const uint32_t* n_contrib,
	// const float* dL_dpixels,
	const float* dL_dpixels_ref_map,
	const float* dL_dpixels_ref_color,
	const float* dL_dpixels_trans_color,
	const float* dL_invdepths,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_drefopacity,
	float* dL_dbeta,
	float* dL_dtranscolors,
	float* dL_drefcolors, 
	// float* dL_dcolors,
	float* dL_dinvdepths)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		conic_ref_opacity,
		colors,
		ref_colors, 
		betas, 
		depths,
		final_Ts,
		final_ref_Ts,
		trans_weights, 
		comp_ref_color, 
		n_contrib,
		// dL_dpixels,
		dL_dpixels_ref_map,
		dL_dpixels_ref_color,
		dL_dpixels_trans_color,
		dL_invdepths,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		// dL_dcolors,
		dL_drefopacity,
		dL_dbeta,
		dL_dtranscolors,
		dL_drefcolors, 
		dL_dinvdepths
		);
}
