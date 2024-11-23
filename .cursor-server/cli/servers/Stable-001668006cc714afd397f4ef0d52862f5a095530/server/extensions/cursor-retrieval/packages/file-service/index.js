/* tslint:disable */
/* eslint-disable */
/* prettier-ignore */

/* auto-generated by NAPI-RS */

const { existsSync, readFileSync } = require('fs')
const { join } = require('path')

const { platform, arch } = process

let nativeBinding = null
let localFileExisted = false
let loadError = null

function isMusl() {
  // For Node 10
  if (!process.report || typeof process.report.getReport !== 'function') {
    try {
      const lddPath = require('child_process').execSync('which ldd').toString().trim()
      return readFileSync(lddPath, 'utf8').includes('musl')
    } catch (e) {
      return true
    }
  } else {
    const { glibcVersionRuntime } = process.report.getReport().header
    return !glibcVersionRuntime
  }
}

switch (platform) {
  case 'android':
    switch (arch) {
      case 'arm64':
        localFileExisted = existsSync(join(__dirname, 'file_service.android-arm64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./file_service.android-arm64.node')
          } else {
            nativeBinding = require('@anysphere/file-service-android-arm64')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm':
        localFileExisted = existsSync(join(__dirname, 'file_service.android-arm-eabi.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./file_service.android-arm-eabi.node')
          } else {
            nativeBinding = require('@anysphere/file-service-android-arm-eabi')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Android ${arch}`)
    }
    break
  case 'win32':
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(
          join(__dirname, 'file_service.win32-x64-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./file_service.win32-x64-msvc.node')
          } else {
            nativeBinding = require('@anysphere/file-service-win32-x64-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'ia32':
        localFileExisted = existsSync(
          join(__dirname, 'file_service.win32-ia32-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./file_service.win32-ia32-msvc.node')
          } else {
            nativeBinding = require('@anysphere/file-service-win32-ia32-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(
          join(__dirname, 'file_service.win32-arm64-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./file_service.win32-arm64-msvc.node')
          } else {
            nativeBinding = require('@anysphere/file-service-win32-arm64-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Windows: ${arch}`)
    }
    break
  case 'darwin':
    localFileExisted = existsSync(join(__dirname, 'file_service.darwin-universal.node'))
    try {
      if (localFileExisted) {
        nativeBinding = require('./file_service.darwin-universal.node')
      } else {
        nativeBinding = require('@anysphere/file-service-darwin-universal')
      }
      break
    } catch {}
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(join(__dirname, 'file_service.darwin-x64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./file_service.darwin-x64.node')
          } else {
            nativeBinding = require('@anysphere/file-service-darwin-x64')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(
          join(__dirname, 'file_service.darwin-arm64.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./file_service.darwin-arm64.node')
          } else {
            nativeBinding = require('@anysphere/file-service-darwin-arm64')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on macOS: ${arch}`)
    }
    break
  case 'freebsd':
    if (arch !== 'x64') {
      throw new Error(`Unsupported architecture on FreeBSD: ${arch}`)
    }
    localFileExisted = existsSync(join(__dirname, 'file_service.freebsd-x64.node'))
    try {
      if (localFileExisted) {
        nativeBinding = require('./file_service.freebsd-x64.node')
      } else {
        nativeBinding = require('@anysphere/file-service-freebsd-x64')
      }
    } catch (e) {
      loadError = e
    }
    break
  case 'linux':
    switch (arch) {
      case 'x64':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'file_service.linux-x64-musl.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./file_service.linux-x64-musl.node')
            } else {
              nativeBinding = require('@anysphere/file-service-linux-x64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'file_service.linux-x64-gnu.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./file_service.linux-x64-gnu.node')
            } else {
              nativeBinding = require('@anysphere/file-service-linux-x64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'arm64':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'file_service.linux-arm64-musl.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./file_service.linux-arm64-musl.node')
            } else {
              nativeBinding = require('@anysphere/file-service-linux-arm64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'file_service.linux-arm64-gnu.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./file_service.linux-arm64-gnu.node')
            } else {
              nativeBinding = require('@anysphere/file-service-linux-arm64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'arm':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'file_service.linux-arm-musleabihf.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./file_service.linux-arm-musleabihf.node')
            } else {
              nativeBinding = require('@anysphere/file-service-linux-arm-musleabihf')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'file_service.linux-arm-gnueabihf.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./file_service.linux-arm-gnueabihf.node')
            } else {
              nativeBinding = require('@anysphere/file-service-linux-arm-gnueabihf')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'riscv64':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'file_service.linux-riscv64-musl.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./file_service.linux-riscv64-musl.node')
            } else {
              nativeBinding = require('@anysphere/file-service-linux-riscv64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'file_service.linux-riscv64-gnu.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./file_service.linux-riscv64-gnu.node')
            } else {
              nativeBinding = require('@anysphere/file-service-linux-riscv64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 's390x':
        localFileExisted = existsSync(
          join(__dirname, 'file_service.linux-s390x-gnu.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./file_service.linux-s390x-gnu.node')
          } else {
            nativeBinding = require('@anysphere/file-service-linux-s390x-gnu')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Linux: ${arch}`)
    }
    break
  default:
    throw new Error(`Unsupported OS: ${platform}, architecture: ${arch}`)
}

if (!nativeBinding) {
  if (loadError) {
    throw loadError
  }
  throw new Error(`Failed to load native binding`)
}

const { DiffType, DiffClient, MerkleClient, GitClient, CommitFileStatus, CommitChainGetFiles } = nativeBinding

module.exports.DiffType = DiffType
module.exports.DiffClient = DiffClient
module.exports.MerkleClient = MerkleClient
module.exports.GitClient = GitClient
module.exports.CommitFileStatus = CommitFileStatus
module.exports.CommitChainGetFiles = CommitChainGetFiles