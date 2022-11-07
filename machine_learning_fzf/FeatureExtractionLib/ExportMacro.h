#pragma once

#ifdef FEATURE_DLL_EXPORTS
	#ifndef FEATURE_DLL_API
		#define FEATURE_DLL_API __declspec(dllexport) 
	#endif
#else
	#ifndef FEATURE_DLL_API
		#define FEATURE_DLL_API __declspec(dllimport) 
	#endif
#endif